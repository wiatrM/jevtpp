#pragma once

#include <nlohmann/json.hpp>
#include <jevt/experience.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <jevt/skill.hpp>

namespace mario {
// Episode-local learning, not neural training. Priors are explicit and every
// forecast carries uncertainty. RAM disappearance is NEVER a confirmed kill.
class AdaptiveAgent {
    using J = nlohmann::json;
    struct Mean {
        int n=0; double mean=0;
        void add(double x) { ++n; mean += (x-mean)/std::min(n,256); }
        J json() const { return {{"samples",n},{"mean",n ? J(mean) : J(nullptr)}}; }
    };
    struct Track {
        int id=0,slot=-1,kind=-1,seen=0,ram_state=0; double x=0,y=0,vx=0,vy=0;
        bool visible=false,velocity=false,stompable=false;
        std::string status="observed";
    };
    std::map<int,Track> tracks_;
    std::map<std::string,Mean> stats_;
    std::map<std::string,int> failures_;
    std::map<std::string,int> minimum_hold_;
    std::map<std::string,jevt::experience::shadow_mean> shadow_physics_;
    jevt::experience::outcome_memory experience_;
    struct Forecast { int due,horizon; double x,y; std::string action; };
    std::vector<Forecast> forecasts_;
    std::map<int,Mean> horizon_errors_;
    int rejected_transitions_=0, cancelled_forecasts_=0;
    J previous_, prediction_, outcome_=J::object(), flight_=J::object(), last_flight_=J::object();
    J committed_landing_=J::object();
    int committed_landing_frame_=-1;
    int run_=-1,life_=-1,world_=-1,stage_=-1,frame_=-1,next_id_=1,target_=-1;
    int archived_unresolved_=0,confirmed_=0,samples_=0,death_count_=0;
    int paired_hazard_phase_=0,paired_hazard_started_=-1,paired_hazard_jump_=0;
    int jump_hold_=0,target_started_=0;
    int attack_target_=-1,attack_frames_=0;
    int task_budget_target_=-1,task_active_frames_=0,task_budget_frame_=-1;
    double task_progress_x_=0,task_progress_distance_=1e9;
    bool task_was_active_=false;
    bool attack_airborne_=false;
    bool jump_cut_=false;
    std::string previous_executed_="noop", preceding_input_="noop", skill_="none",plan_key_="none",task_status_="exploring";
    Mean prediction_error_;
    static double num(const J& j,const char* k,double fallback=0) {
        auto it=j.find(k); return it!=j.end() && it->is_number() && std::isfinite(it->get<double>()) ? it->get<double>() : fallback;
    }
    static bool yes(const J& j,const char* k) { return j.value(k,J(nullptr))==true; }
    static bool jumping(const std::string& a) { return a.find("jump")!=std::string::npos; }
    static bool platform_arena(const J& s) {
        const auto p=s.value("player",J::object());
        for(const auto& surface:s.value("collision",J::object()).value("landing_surfaces",J::array()))
            if(num(surface,"right_x")>num(p,"x")+16 && num(surface,"left_x")<num(p,"x")+176 &&
                num(surface,"y")>=num(p,"feet_y")-96 && num(surface,"y")<208) return true;
        return false;
    }
    static J next_platform(const J& s) {
        const auto p=s.value("player",J::object());J target=nullptr;
        for(const auto& surface:s.value("collision",J::object()).value("landing_surfaces",J::array()))
            if(num(surface,"left_x")>num(p,"x")+12 && num(surface,"y")>=num(p,"feet_y")-96 &&
                num(surface,"y")<208 && (target.is_null() || num(surface,"left_x")<num(target,"left_x"))) target=surface;
        return target;
    }
    static J support_at(const J& s,double x,double feet) {
        for(const auto& surface:s.value("collision",J::object()).value("landing_surfaces",J::array()))
            if(std::abs(num(surface,"y")-feet)<3 && x+10>num(surface,"left_x") && x+2<num(surface,"right_x")) return surface;
        return nullptr;
    }
    static J following_support(const J& s,const J& first) {
        J result=nullptr;if(!first.is_object()) return result;
        for(const auto& surface:s.value("collision",J::object()).value("landing_surfaces",J::array()))
            if(num(surface,"left_x")>=num(first,"right_x") && num(surface,"y")>=num(first,"y")-96 &&
                (result.is_null() || num(surface,"left_x")<num(result,"left_x"))) result=surface;
        return result;
    }
    static J attack_corridor(const J& s,const Track& target) {
        const auto p=s.value("player",J::object()),c=s.value("collision",J::object());
        const double px=num(p,"x")+8,ex=target.x+8,feet=num(p,"feet_y");
        if(std::abs(ex-px)<16) return {{"blocked",false},{"source","current_RAM_body_corridor"}};
        for(const auto& col:c.value("planning_columns",c.value("columns",J::array()))) {
            const double cx=num(col,"x")+8;
            if(cx<=std::min(px,ex)+4 || cx>=std::max(px,ex)-4) continue;
            const double f=(cx-px)/(ex-px),bottom=feet+f*(target.y+24-feet);
            for(const auto& y:col.value("solid_y",J::array()))
                if(y.get<double>()<bottom-2 && y.get<double>()+16>bottom-16)
                    return {{"blocked",true},{"block_x",num(col,"x")},{"block_y",y},
                            {"source","current_RAM_body_corridor"},{"requires","navigate around blocks before attack"}};
        }
        return {{"blocked",false},{"source","current_RAM_body_corridor"},
                {"limitation","Clear direct corridor is not proof that a stomp is reachable"}};
    }
    static std::string power(const J& p) { return p.value("powerup_status",std::string("small")); }
    static std::string context(const J& p,const std::string& a) {
        return power(p)+":"+(yes(p,"grounded")?"ground:":"air:")+a;
    }
    // Relative, bounded features: translating a scene preserves its identity,
    // but a different support, target, speed or blocked approach does not.
    static std::string failure_key(const J& s,const std::string& action,const J& target=J(nullptr)) {
        const auto p=s.value("player",J::object()),c=s.value("collision",J::object());
        const auto bin=[](double value,double width) {
            return std::to_string(static_cast<int>(std::floor(std::clamp(value,-1024.,1024.)/width)));
        };
        std::map<std::string,std::string> f{{"schema","relative-v1"},{"action",action},{"size",power(p)},
            {"phase",yes(p,"grounded")?"ground":"air"},{"vx",bin(num(p,"physics_vx",num(p,"vx")),1.)},
            {"vy",bin(num(p,"vy"),2.)},{"wall",bin(num(c,"obstacle_distance_pixels",999),16.)},
            {"gap",bin(num(c,"gap_distance_pixels",999),16.)},{"support","unknown"},{"target","unknown"}};
        for(const auto& col:c.value("planning_columns",c.value("columns",J::array()))) {
            const double dx=num(col,"x")-num(p,"x");
            if(dx < -16 || dx > 64) continue;
            std::string occupied;
            for(const auto& y:col.value("solid_y",J::array())) {
                const double dy=y.get<double>()-num(p,"feet_y");
                if(dy>=-64 && dy<=32) occupied+=bin(dy,16.)+",";
            }
            f["local_col_"+bin(dx,16.)]=occupied.empty()?"empty":occupied;
        }
        for(const auto& support:c.value("landing_surfaces",J::array()))
            if(num(support,"left_x")<=num(p,"x")+8 && num(support,"right_x")>=num(p,"x")+8 &&
                std::abs(num(support,"y")-num(p,"feet_y"))<=3)
                f["support"]=bin(num(support,"right_x")-num(support,"left_x"),16.)+":"+
                    bin(num(support,"right_x")-num(p,"x")-8,8.);
        if(target.is_object()) f["target"]=bin(num(target,"left_x",num(target,"x"))-num(p,"x"),16.)+":"+
            bin(num(target,"right_x",num(target,"x")+16)-num(target,"left_x",num(target,"x")),16.)+":"+
            bin(num(target,"y",num(target,"screen_y"))-num(p,"feet_y"),16.);
        return jevt::experience::context_key(f);
    }
    int contextual_failure_count(const J& s,const std::string& action,const J& target=J(nullptr)) const {
        const auto* evidence=experience_.find(failure_key(s,action,target));
        return evidence?static_cast<int>(evidence->failures):0;
    }
    void learn_parameter(const std::string& key,double value,double prior,double lo,double hi) {
        stats_[key].add(value);
        shadow_physics_.try_emplace(key,lo,hi).first->second.observe(value,prior);
    }
    double estimate(const std::string& key,double prior,double lo,double hi) const {
        auto it=shadow_physics_.find(key);
        return std::clamp(it==shadow_physics_.end()?prior:it->second.value_or(prior),lo,hi);
    }
    struct Body { double x,y,vx,vy; bool ground,a; bool cut=false; double air_cap=3.; };
    struct Motion { Body body; bool ceiling=false,wall=false,unknown=false,landed=false; };
    struct Tile { double x,y; };
    struct Geometry { std::vector<Tile> tiles; std::vector<double> columns; };
    static Geometry geometry(const J& columns) {
        Geometry g;
        for(const auto& col:columns) {
            const double x=num(col,"x");g.columns.push_back(x);
            for(const auto& y:col.value("solid_y",J::array())) g.tiles.push_back({x,y.get<double>()});
        }
        return g;
    }
    Motion advance(Body b,const std::string& action,const Geometry& tiles,const std::string& size) const {
        const double oldx=b.x,oldy=b.y;
        const bool a=jumping(action), run=action.find("run")!=std::string::npos;
        const int dir=action.find("left")!=std::string::npos?-1:action.find("right")!=std::string::npos?1:0;
        const std::string key=size+":"+(b.ground?"ground:":"air:")+action;
        double ax=dir*(run?.14:.1);
        if(dir) ax=estimate(key+":ax",ax,-.5,.5);
        else ax=!b.ground?0:b.vx>0?-.08:b.vx<0?.08:0;
        const double limit=b.ground?(run?3.:1.75):b.air_cap;
        b.vx=dir ? (dir*b.vx>limit ? b.vx-(b.ground?dir*.08:0) : std::clamp(b.vx+ax,-limit,limit)) :
                   (std::abs(b.vx)<.08?0:b.vx+ax);
        if(b.ground && a && !b.a) {
            b.vy=-estimate(size+":jump_impulse",6.,3.,8.); b.ground=false;b.cut=false;
            b.air_cap=std::max(1.75,std::abs(b.vx));
        } else if(!b.ground) {
            // Re-pressing A cannot restore a jump already shortened by release.
            b.cut|=!a;
            const bool held=a && !b.cut && b.vy<0;
            const auto gravity_key=size+(held?":gravity_hold":":gravity_release");
            b.vy=std::min(5.,b.vy+estimate(gravity_key,held?.25:.65,.12,1.2));
        } else b.vy=0;
        b.a=a;b.x+=b.vx;
        Motion m; const double height=size=="small"?16:32;
        // Collision sweeps use only observed tiles, never a memorized route.
        for(const auto& tile:tiles.tiles) {
            const double tx=tile.x,ty=tile.y;
            if(oldy>ty+1 && oldy-height<ty+15 && b.x+12>tx && b.x<tx+16) {
                if(b.vx>0 && oldx+12<=tx+1) { b.x=tx-12;b.vx=0;m.wall=true; }
                if(b.vx<0 && oldx>=tx+15) { b.x=tx+16;b.vx=0;m.wall=true; }
            }
        }
        b.y+=b.vy; bool support=false,known=false;
        for(const double tx:tiles.columns) if(b.x+10>tx && b.x+2<tx+16) known=true;
        for(const auto& tile:tiles.tiles) {
            const double tx=tile.x;
            if(b.x+10<=tx || b.x+2>=tx+16) continue;
                const double ty=tile.y;
                if(b.vy>=0 && oldy<=ty+2 && b.y>=ty) {
                    b.y=ty;b.vy=0;support=true;m.landed=!b.ground;b.ground=true;
                } else if(b.vy<0 && oldy-height>=ty+15 && b.y-height<ty+16) {
                    b.y=ty+16+height;b.vy=0;m.ceiling=true;
                }
                if(b.ground && std::abs(b.y-ty)<2) support=true;
        }
        if(b.ground && !support) b.ground=false;
        m.unknown=!known;m.body=b;return m;
    }
    static J track_json(const Track& t,int frame) {
        return {{"id",t.id},{"slot",t.slot},{"kind_id",t.kind},{"x",t.x},{"screen_y",t.y},
            {"vx",t.vx},{"velocity_observed",t.velocity},{"stompable",t.stompable},
            {"visible",t.visible},{"age_frames",frame-t.seen},{"status",t.status},{"ram_state",t.ram_state},
            {"uncertainty",t.status=="confirmed_eliminated"?"defeat_confirmed":t.visible?(t.velocity?"position_and_velocity_observed":"velocity_unknown"):"unobserved_not_eliminated"}};
    }
public:
    void reset() { *this=AdaptiveAgent{}; }
    J observe(const J& s,int input_frame) {
        const auto session=s.value("session",J::object()), ep=s.value("episode",J::object());
        const auto level=s.value("level",J::object()),p=s.value("player",J::object());
        const int run=num(session,"run_id",0),frame=num(session,"session_frame",input_frame);
        const int world=num(level,"world",num(session,"world",1)),stage=num(level,"stage",num(session,"stage",1));
        const int life=num(session,"deaths",0);
        if(run_!=-1 && run_!=run) reset();
        if(yes(session,"game_over") || yes(ep,"game_over")) {
            reset();run_=run;return {{"accepted",false},{"reset_reason","game_over"},{"events",J::array()}};
        }
        const bool boundary=world_!=world || stage_!=stage || life_!=life || (frame_>=0 && frame<frame_);
        const bool death= yes(ep,"dead") || yes(session,"life_lost") || (life_>=0 && life>life_);
        if(!boundary && frame==frame_) return {{"accepted",false},{"events",J::array()},{"reason","duplicate_physical_frame"}};
        J events=J::array();
        if(death && (!previous_.empty()) && !yes(previous_.value("episode",J::object()),"dead") &&
            !yes(previous_.value("session",J::object()),"life_lost")) {
            ++death_count_;
            const std::string credited=flight_.empty()?plan_key_:flight_.at("signature").get<std::string>();
            const std::string credit_key=flight_.value("credit_key",failure_key(previous_,credited));
            const std::string failure_class=!flight_.empty() && !yes(flight_,"takeoff_confirmed")?
                "takeoff_not_observed":"life_lost_cause_unverified";
            experience_.record(credit_key,std::to_string(run)+":"+std::to_string(frame),
                jevt::experience::outcome::failed,failure_class);
            ++failures_[credited];
            if(!flight_.empty()) {
                const auto target=flight_.value("target",J(nullptr));
                flight_["failure_hypothesis"]=failure_class;
                if(yes(flight_,"takeoff_confirmed") && !yes(flight_,"contact_suspected") &&
                    target.is_object() && num(flight_,"last_feet_y")>num(target,"y")+24 &&
                    num(flight_,"last_x")<num(target,"left_x")+8) {
                    const auto key=flight_.value("context",credit_key);
                    minimum_hold_[key]=std::max(minimum_hold_[key],std::min(32,static_cast<int>(num(flight_,"held_frames"))+6));
                    flight_["failure_hypothesis"]="undershot_observed_support";
                    flight_["next_minimum_hold_frames"]=minimum_hold_[key];
                }
                flight_["result"]="life_lost_before_verified_landing";
                flight_["end_frame"]=frame;last_flight_=flight_;flight_=J::object();
            }
            const auto prior=previous_.value("player",J::object());
            const bool falling=num(prior,"feet_y")>232;
            events.push_back({{"type","life_lost"},{"skill",skill_},{"plan_signature",credited},{"cause",falling?"fall_hypothesis":"unknown_collision_or_timeout"},
                {"context_key",credit_key},{"learning_class",failure_class},
                {"certainty","life_loss_confirmed_cause_not_proven"}});
        }
        if(boundary) {
            for(auto& [id,t]:tracks_) if(t.status!="confirmed_eliminated") ++archived_unresolved_;
            tracks_.clear();target_=-1;previous_=J();prediction_=J();jump_hold_=0;
            flight_=J::object();
            cancelled_forecasts_+=static_cast<int>(forecasts_.size());forecasts_.clear();
            attack_target_=-1;attack_frames_=0;attack_airborne_=false;
            jump_cut_=false;
            committed_landing_=J::object();committed_landing_frame_=-1;
        }
        run_=run;world_=world;stage_=stage;life_=life;
        const auto actual=s.value("recent_control",J::object()).value("action",J(nullptr));
        if(!flight_.empty() && !flight_.contains("credit_key")) {
            flight_["credit_key"]=failure_key(previous_.empty()?s:previous_,flight_.value("signature",plan_key_),flight_.value("target",J(nullptr)));
            flight_["context"]=flight_["credit_key"];
        }
        if(flight_.value("target",J(nullptr)).is_object()) {
            auto& target=flight_["target"];
            for(const auto& surface:s.value("collision",J::object()).value("landing_surfaces",J::array()))
                if(num(surface,"y")==num(target,"y") && num(surface,"left_x")<num(target,"right_x") &&
                    num(surface,"right_x")>num(target,"left_x")) target=surface;
        }
        const bool contiguous=!previous_.empty() && frame==frame_+1 && actual.is_string() && !death &&
            !yes(ep,"stage_clear") && !yes(session,"stage_transition") &&
            p.value("velocity_valid",true) && previous_.at("player").value("velocity_valid",true) &&
            power(p)==power(previous_.at("player")) &&
            std::abs(num(p,"x")-num(previous_.at("player"),"x"))<16 &&
            std::abs(num(p,"feet_y")-num(previous_.at("player"),"feet_y"))<16 &&
            !yes(previous_.value("episode",J::object()),"dead") && !yes(previous_.value("session",J::object()),"life_lost");
        if(contiguous) {
            const auto pp=previous_.at("player");
            const auto previous_input=previous_.value("recent_control",J::object()).value("action",J(nullptr));
            const auto action=previous_input.is_string()?previous_input.get<std::string>():actual.get<std::string>();
            if(yes(p,"grounded")) jump_cut_=false;
            else if(!jumping(action)) jump_cut_=true;
            const auto key=context(pp,action); const double dx=num(p,"x")-num(pp,"x");
            const double dy=num(p,"feet_y")-num(pp,"feet_y");
            if(std::abs(dx)<16 && std::abs(dy)<16) {
                ++samples_;stats_[key+":dx"].add(dx);stats_[key+":dy"].add(dy);
                const double ax=num(p,"physics_vx",num(p,"vx"))-num(pp,"physics_vx",num(pp,"vx"));
                const double speed_limit=action.find("run")!=std::string::npos?3.:1.75;
                const double old_vx=num(pp,"physics_vx",num(pp,"vx"));
                Body prior_body{num(pp,"x"),num(pp,"feet_y"),old_vx,-num(pp,"vy"),yes(pp,"grounded"),jumping(preceding_input_)};
                prior_body.cut=jump_cut_;prior_body.air_cap=std::max(1.75,std::abs(old_vx));
                const auto contact=advance(prior_body,action,geometry(previous_.value("collision",J::object()).value("planning_columns",J::array())),power(pp));
                const bool contact_transition=contact.wall || contact.ceiling || contact.landed ||
                    (!yes(pp,"grounded") && yes(p,"grounded")) ||
                    (num(pp,"vy")>1 && num(p,"vy")==0) || (num(pp,"vy")<0 && num(p,"vy")>1);
                if(contact_transition) ++rejected_transitions_;
                const bool braking=(action.find("left")!=std::string::npos && old_vx>0) ||
                    (action.find("right")!=std::string::npos && old_vx<0);
                if(!contact_transition && yes(pp,"grounded")==yes(p,"grounded") && std::abs(ax)<.6 && std::abs(dx)>.1 &&
                    (braking || std::abs(old_vx)<speed_limit-.15))
                    learn_parameter(key+":ax",ax,(action.find("left")!=std::string::npos?-1:1)*(action.find("run")!=std::string::npos?.14:.1),-.5,.5);
                const double vy=num(p,"vy"),oldvy=num(pp,"vy");
                if(yes(pp,"grounded") && !yes(p,"grounded") && vy>1 && jumping(action))
                    learn_parameter(power(p)+":jump_impulse",vy,6.,3.,8.);
                // Integer pixel velocities alternate between zero and one
                // acceleration. Excluding zero would learn gravity=1 always.
                if(!contact_transition && !yes(pp,"grounded") && !yes(p,"grounded") && oldvy-vy>=-1 && oldvy-vy<=2 && vy>-5) {
                    const bool held=jumping(action)&&!jump_cut_&&oldvy>0;
                    learn_parameter(power(p)+(held?":gravity_hold":":gravity_release"),oldvy-vy,held?.25:.65,.12,1.2);
                }
                if(prediction_.is_object() && actual.get<std::string>()==previous_executed_) {
                    const double error=std::hypot(num(p,"x")-num(prediction_,"x"),num(p,"feet_y")-num(prediction_,"feet_y"));
                    prediction_error_.add(error);
                    events.push_back({{"type","prediction_checked"},{"error_px",error},{"action",action}});
                }
                if(!flight_.empty() && !yes(p,"grounded")) {
                    if(!yes(flight_,"takeoff_confirmed") && vy>1 && dy<0 &&
                        frame-num(flight_,"start_frame")<=8) {
                        flight_["takeoff_confirmed"]=true;flight_["takeoff_frame"]=frame;
                        events.push_back({{"type","takeoff_confirmed"},{"source","observed_upward_displacement_and_velocity"}});
                    }
                    if(contact_transition) {
                        flight_["contact_suspected"]=true;
                        flight_["contact_source"]="predicted_collision_or_velocity_discontinuity_guard";
                    }
                    if(jumping(action) && !jump_cut_) flight_["held_frames"]=static_cast<int>(num(flight_,"held_frames"))+1;
                    flight_["airborne_observed"]=true;
                    flight_["apex_feet_y"]=std::min(num(flight_,"apex_feet_y"),num(p,"feet_y"));
                    flight_["last_x"]=num(p,"x");flight_["last_feet_y"]=num(p,"feet_y");
                }
                if(!yes(pp,"grounded") && yes(p,"grounded")) {
                    events.push_back({{"type","landing_confirmed"},{"skill",skill_}});
                    if(!flight_.empty()) {
                        flight_["result"]="landing_confirmed";flight_["end_frame"]=frame;
                        flight_["landed_x"]=num(p,"x");flight_["landed_feet_y"]=num(p,"feet_y");
                        const auto target=flight_.value("target",J(nullptr));
                        const bool reached=target.is_object() && std::abs(num(p,"feet_y")-num(target,"y"))<3 &&
                            num(p,"x")+6>=num(target,"left_x") && num(p,"x")+6<=num(target,"right_x");
                        flight_["target_reached"]=reached;
                        // Landing elsewhere is not task success; absence of a
                        // target verifier remains unknown, not a fabricated win.
                        experience_.record(flight_.value("credit_key",std::string()),
                            std::to_string(run)+":"+std::to_string(static_cast<int>(num(flight_,"start_frame"))),
                            reached && yes(flight_,"takeoff_confirmed")?jevt::experience::outcome::succeeded:jevt::experience::outcome::unknown);
                        last_flight_=flight_;flight_=J::object();
                    }
                }
            }
        }
        if(!contiguous && !previous_.empty()) ++rejected_transitions_;
        const std::string current_action=actual.is_string()?actual.get<std::string>():"";
        for(auto it=forecasts_.begin();it!=forecasts_.end();) {
            if(!contiguous || it->action!=current_action || frame>it->due) {
                ++cancelled_forecasts_;it=forecasts_.erase(it);
            } else if(frame==it->due) {
                horizon_errors_[it->horizon].add(std::hypot(num(p,"x")-it->x,num(p,"feet_y")-it->y));
                it=forecasts_.erase(it);
            } else ++it;
        }
        // Forecasts are frozen now, then checked against future observations.
        // Holding the action is a condition, not an assumption about execution.
        if(frame%4==0 && actual.is_string() && !death && p.value("velocity_valid",true)) {
            Body body{num(p,"x"),num(p,"feet_y"),num(p,"physics_vx",num(p,"vx")),-num(p,"vy"),yes(p,"grounded"),jumping(preceding_input_)};
            body.cut=jump_cut_;body.air_cap=std::max(1.75,std::abs(body.vx));
            const auto tiles=geometry(s.value("collision",J::object()).value("planning_columns",J::array()));
            for(int h=1;h<=32;++h) {
                const auto forecast=advance(body,current_action,tiles,power(p));body=forecast.body;
                if(forecast.unknown) break;
                if(h==1 || h==4 || h==8 || h==16 || h==32) forecasts_.push_back({frame+h,h,body.x,body.y,current_action});
            }
        }
        for(auto& [id,t]:tracks_) {t.visible=false;if(t.status!="confirmed_eliminated") t.status="unobserved";}
        const auto hazard=s.value("hazard",J::object());
        for(const auto& e:hazard.value("nearby_enemies",hazard.value("upcoming_enemies",J::array()))) {
            if(!e.contains("x") || !e.contains("slot") || !e.contains("kind_id")) continue;
            const int slot=num(e,"slot"),kind=num(e,"kind_id"); const double x=num(e,"x");
            int match=-1;
            for(auto& [id,t]:tracks_) if(t.slot==slot && t.kind==kind && t.status!="confirmed_eliminated" &&
                frame-t.seen<=90 && std::abs(x-t.x)<=std::min(96.,16.+4.*(frame-t.seen))) { match=id;break; }
            if(match<0) {
                if(tracks_.size()>=128) {
                    auto it=std::min_element(tracks_.begin(),tracks_.end(),[](const auto& a,const auto& b){return a.second.seen<b.second.seen;});
                    if(it->second.status!="confirmed_eliminated") ++archived_unresolved_;
                    if(target_==it->first) target_=-1;
                    tracks_.erase(it);
                }
                match=next_id_++;tracks_[match].id=match;
                events.push_back({{"type","entity_discovered"},{"id",match}});
            }
            auto& t=tracks_[match];t.slot=slot;t.kind=kind;t.x=x;t.y=num(e,"screen_y");
            t.ram_state=num(e,"state");
            t.seen=frame;t.visible=true;t.stompable=yes(e,"stompable");t.velocity=yes(e,"velocity_observed");
            t.vx=num(e,"relative_velocity_x")+num(p,"vx");t.status="observed";
            t.vy=t.velocity?num(e,"vy"):0;
        }
        // Identity-specific alive -> defeated evidence comes from the RAM adapter.
        // A score increase or aggregate kill count is not enough for attribution.
        for(const auto& e:hazard.value("elimination_events",J::array())) {
            for(auto& [id,t]:tracks_) if(t.slot==num(e,"slot",-1) && t.kind==num(e,"kind_id",-1) &&
                std::abs(t.x-num(e,"x"))<32 && frame-t.seen<=8 && t.status!="confirmed_eliminated") {
                t.status="confirmed_eliminated";t.visible=false;++confirmed_;
                events.push_back({{"type","elimination_confirmed"},{"id",id},{"source","RAM_alive_to_defeated"}});
            }
        }
        jump_hold_=actual.is_string() && jumping(actual.get<std::string>())?jump_hold_+1:0;
        const auto old_input=previous_.is_object()?previous_.value("recent_control",J::object()).value("action",J(nullptr)):J(nullptr);
        preceding_input_=old_input.is_string()?old_input.get<std::string>():"noop";
        previous_=s;frame_=frame;
        outcome_={{"accepted",true},{"events",events},{"dynamics_samples",samples_},{"deaths_this_run",death_count_},
            {"credit_assignment","observed results; death causes remain hypotheses"}};
        return outcome_;
    }
    J world() const {
        J entities=J::array(); int unresolved=archived_unresolved_;
        for(const auto& [id,t]:tracks_) {entities.push_back(track_json(t,frame_));if(t.status!="confirmed_eliminated") ++unresolved;}
        return {{"entities",entities},{"confirmed_eliminations",confirmed_},{"unresolved",unresolved},
            {"archived_unresolved",archived_unresolved_},{"scope","current_run_observed_entities"},
            {"unknown_world","not explored; not assumed empty"},{"identity","slot + type + spatial continuity + life/stage generation"}};
    }
    J dynamics() const {
        J stats=J::object();for(const auto& [key,value]:stats_) stats[key]=value.json();
        J supervisor=J::object(),horizons=J::object(),evidence=J::object();
        for(const auto& [key,estimator]:shadow_physics_) supervisor[key]={{"state",estimator.promoted()?"promoted":"shadow"},
            {"training_samples",estimator.samples()},{"paired_checks",estimator.checks()},
            {"prior_mae",estimator.checks()?J(estimator.prior_error()):J(nullptr)},
            {"candidate_mae",estimator.checks()?J(estimator.candidate_error()):J(nullptr)}};
        for(int h:{1,4,8,16,32}) {
            const auto it=horizon_errors_.find(h);
            horizons[std::to_string(h)]=it==horizon_errors_.end()?Mean{}.json():it->second.json();
        }
        for(const auto& [key,value]:experience_.records()) evidence[key]={{"successes",value.successes},
            {"failures",value.failures},{"unknown",value.unknown},{"failure_classes",value.failure_classes}};
        return {{"samples",samples_},{"conditioned_statistics",stats},{"prediction_error_px",prediction_error_.json()},
            {"learning_supervisor",supervisor},{"contextual_evidence",evidence},
            {"horizon_prediction_error_px",horizons},{"cancelled_forecasts",cancelled_forecasts_},
            {"rejected_parameter_transitions",rejected_transitions_},
            {"forecast_contract","prequential; frozen geometry and same actual input; changed action/boundary invalidates forecast"},
            {"promotion_contract","32 paired future checks, >=24 wins, >=10% MAE improvement; reversible; not a policy safety guarantee"},
            {"active_flight",flight_},{"last_flight",last_flight_},
            {"learned_minimum_hold_frames",minimum_hold_},
            {"jump_hold_frames",jump_hold_},{"jump_cut",jump_cut_},{"input_delay_frames",1},{"source","online_action_conditioned_physics_with_explicit_priors"},
            {"priors",{{"jump_impulse",6.},{"gravity_hold",.25},{"gravity_release",.65}}},
            {"limitations","Approximate collision model; no calibrated success probabilities or neural updates."}};
    }
    J tasks(const J& s) {
        const auto p=s.value("player",J::object()); const double x=num(p,"x");
        const auto mode=s.value("strategy",J::object()).value("mode",std::string("speedrun"));
        if(target_>=0 && tracks_.contains(target_) && tracks_.at(target_).status=="confirmed_eliminated") target_=-1;
        const bool hunt=mode=="hunter" || mode=="score_attack";
        int deferred=-1;
        if(hunt && target_>=0 && tracks_.contains(target_)) {
            const auto& t=tracks_.at(target_);
            if(!t.visible || !t.stompable || task_active_frames_>360) {
                for(const auto& [id,other]:tracks_) if(id!=target_ && other.visible && other.stompable && other.status!="confirmed_eliminated") {
                    deferred=target_;target_=-1;break;
                }
            }
        }
        if(hunt && target_<0) {
            double best=1e9;
            for(const auto& [id,t]:tracks_) if(id!=deferred && t.visible && t.status!="confirmed_eliminated" && std::abs(t.x-x)<192 && std::abs(t.x-x)<best) {
                target_=id;best=std::abs(t.x-x);
            }
            target_started_=frame_;
        }
        J target=nullptr,approach=nullptr;task_status_="exploring";
        if(hunt && target_>=0 && tracks_.contains(target_)) {
            auto& t=tracks_.at(target_);target=track_json(t,frame_);
            approach=attack_corridor(s,t);
            const double distance=std::abs(t.x-x);
            if(task_budget_target_!=target_) {
                task_budget_target_=target_;task_active_frames_=0;
                task_progress_x_=x;task_progress_distance_=distance;task_was_active_=false;
            }
            if(task_was_active_ && task_budget_frame_>=0)
                task_active_frames_+=std::max(0,frame_-task_budget_frame_);
            // Credit observed approach progress, not repeated jump commands or
            // the target merely moving around Mario. Blocked time is excluded.
            if(std::abs(x-task_progress_x_)>=8 && distance<task_progress_distance_-8) {
                task_active_frames_=0;task_progress_x_=x;task_progress_distance_=distance;
            }
            const auto col=s.value("collision",J::object());
            if(!t.visible) task_status_="unresolved_target_unobserved";
            else if(!t.stompable) task_status_="blocked_no_supported_attack";
            else if(t.x<x-16 && !yes(col,"behind_floor_safe")) task_status_="blocked_unsafe_return";
            else if(task_active_frames_>360) task_status_="blocked_attempt_budget";
            else if(yes(approach,"blocked")) task_status_="blocked_observed_blocks";
            else task_status_="active";
        }
        task_budget_frame_=frame_;task_was_active_=task_status_=="active";
        const J steps=approach.is_object() && yes(approach,"blocked")?
            J::array({{{"skill","navigate_around_observed_block"},{"status","requires_feasible_route"},
                       {"block_x",approach.value("block_x",J(nullptr))}},
                      {{"skill","reach_attack_pose"},{"status","waiting_for_navigation"}},
                      {{"skill","stomp_target"},{"status","waiting_for_attack_pose"}},
                      {{"skill","verify_elimination"},{"status","requires_RAM_evidence"}}}):J::array();
        int visible=0,unresolved=archived_unresolved_;
        for(const auto& [id,t]:tracks_) if(t.status!="confirmed_eliminated") {++unresolved;if(t.visible) ++visible;}
        return {{"objective",hunt?"eliminate_observed_hostiles":mode=="collector"?"collect_resources":"advance_stage"},
            {"confirmed_eliminations",confirmed_},{"unresolved_targets",unresolved},{"visible_targets",visible},
            {"clearance",unresolved?"incomplete":confirmed_?"observed_targets_cleared":"not_yet_observed"},
            {"status",task_status_},{"target",target},{"approach",approach},{"target_age_frames",target_>=0?frame_-target_started_:0},
            {"active_attempt_frames",task_active_frames_},{"subtasks",steps},
            {"attempt_budget_semantics","active frames without observed approach progress; blocked time excluded"},
            {"completion",hunt?"not_proven":"in_progress"},{"success_requires","RAM-confirmed outcome, not disappearance"},
            {"conflict_policy","survival first; blocked targets remain unresolved, never count as kills"}};
    }
    J reachability(const J& s,const J& task) const {
        const auto col=s.value("collision",J::object()),p=s.value("player",J::object());
        const auto tiles=col.value("planning_columns",col.value("columns",J::array()));
        J surfaces=J::array();
        for(const auto& c:tiles) for(const auto& sy:c.value("solid_y",J::array())) {
            const int y=sy.get<int>();bool covered=false;
            for(const auto& other:c.at("solid_y")) if(other.get<int>()==y-16) covered=true;
            if(!covered) surfaces.push_back({{"x",c.at("x")},{"y",y},{"width",16}});
        }
        return {{"available",yes(col,"available") && p.contains("x") && p.contains("feet_y") && !tiles.empty()},
            {"tiles",tiles},{"surfaces",surfaces},{"target",task.at("target")},
            {"reachability","candidate-specific swept collision checks; unobserved space is unknown"}};
    }
    J engage(const J& s,const J& task) {
        const auto p=s.value("player",J::object()),c=s.value("collision",J::object());
        const auto ep=s.value("episode",J::object()),session=s.value("session",J::object());
        const auto target=task.at("target");
        J proposal{{"apply",false},{"phase","navigation"},{"action",nullptr},{"target_id",nullptr},
            {"source","closed_loop_target_skill"},{"completion","not_confirmed"}};
        const auto mode=s.value("strategy",J::object()).value("mode",std::string());
        if(!s.value("planner_enabled",true) || mode!="hunter" ||
            !yes(c,"available") || yes(ep,"dead") || yes(ep,"stage_clear") || yes(session,"life_lost") ||
            yes(session,"stage_transition") || yes(session,"game_over") || yes(session,"game_complete")) return proposal;
        if(!target.is_object() || !yes(target,"visible")) {
            if(attack_target_>=0 && tracks_.contains(attack_target_) && tracks_.at(attack_target_).status=="confirmed_eliminated") {
                proposal["phase"]="elimination_verified";proposal["completion"]="RAM_confirmed";proposal["target_id"]=attack_target_;
            }
            attack_target_=-1;attack_frames_=0;attack_airborne_=false;return proposal;
        }
        // Task vetoes apply in flight too. Previously this skill ignored both
        // a 360-frame attempt timeout and walls whenever Mario was airborne.
        if(task.value("status",std::string())!="active") {
            proposal["phase"]=task.at("status");proposal["target_id"]=target.at("id");
            proposal["approach"]=task.value("approach",J(nullptr));
            attack_frames_=0;attack_airborne_=false;return proposal;
        }
        const double dx=num(target,"x")-num(p,"x"),vx=num(p,"physics_vx",num(p,"vx"));
        const double evx=yes(target,"velocity_observed")?num(target,"vx"):0;
        const double top=num(target,"screen_y")+8,feet=num(p,"feet_y");
        const bool ground=yes(p,"grounded");const int id=num(target,"id",-1);
        proposal["target_id"]=id;
        const auto affordance=s.value("knowledge_affordances",J::object());
        const auto level=s.value("level",J::object());
        const auto current_stage=std::to_string(level.value("world",0))+"-"+std::to_string(level.value("stage",0));
        for(const auto& zone:affordance.value("death_zones",J::array())) {
            if(!zone.is_object() || zone.value("stage",std::string())!=current_stage ||
                zone.value("unsafe_waits",0)<1)continue;
            bool kind_matches=false;
            for(const auto& kind:zone.value("target_kinds",J::array()))
                kind_matches|=kind.is_number_integer() && kind.get<int>()==num(target,"kind_id",-1);
            if(!kind_matches)continue;
            const double distance=num(zone,"x")-num(p,"x");
            if(ground && yes(c,"behind_floor_safe") && dx>0 && dx<96 &&
                std::abs(top-feet)>40 && distance>=-32 &&
                distance<=num(zone,"warning_radius_px",112)) {
                proposal["apply"]=true;
                proposal["phase"]="retreat_from_failed_wait";
                proposal["action"]="left";
                proposal["knowledge_evidence"]={{"stage",current_stage},
                    {"x",zone.value("x",0)},{"events",zone.value("unsafe_waits",0)},
                    {"source","confirmed_post_step_life_loss"}};
                attack_frames_=0;attack_airborne_=false;
                return proposal;
            }
        }
        // A confirmed loss at this location is feedback, not a causal claim
        // about the enemy.  Replan the airborne intercept instead of repeating
        // the same braking/noop trajectory into the same landing collision.
        for(const auto& zone:affordance.value("death_zones",J::array())) {
            if(!zone.is_object() || zone.value("stage",std::string())!=current_stage ||
                zone.value("unsafe_intercepts",0)<1) continue;
            bool kind_matches=false;
            for(const auto& kind:zone.value("target_kinds",J::array()))
                kind_matches|=kind.is_number_integer() && kind.get<int>()==num(target,"kind_id",-1);
            if(!kind_matches)continue;
            const double distance=num(zone,"x")-num(p,"x");
            if(!ground && dx>8 && dx<96 && distance>=-32 &&
                distance<=num(zone,"warning_radius_px",112)) {
                proposal["phase"]="observed_death_zone_replan";
                proposal["knowledge_evidence"]={{"stage",current_stage},
                    {"x",zone.value("x",0)},{"events",zone.value("events",0)},
                    {"source","confirmed_post_step_life_loss"}};
                attack_frames_=0;attack_airborne_=false;
                return proposal;
            }
        }
        if(attack_target_!=id) {attack_target_=id;attack_frames_=0;attack_airborne_=false;}
        if(std::abs(dx)>160 || num(c,"gap_distance_pixels",999)<48 ||
            (dx< -16 && !yes(c,"behind_floor_safe"))) {
            proposal["phase"]="blocked_geometry";return proposal;
        }
        if(!yes(target,"stompable")) {proposal["phase"]="blocked_attack_capability";return proposal;}
        auto steer=[&](double desired,bool a) -> std::string {
            const auto direction=vx>desired+.12?"left":vx<desired-.12?"right":"noop";
            return a?(std::string(direction)=="noop"?"jump":std::string(direction)+"_jump"):direction;
        };
        std::string action,phase;
        if(ground && std::abs(top-feet)>40) {
            // A walking enemy on an overhead ledge can descend into reach.
            // Hold the encounter instead of silently running past that ledge.
            action=steer(dx>88?.6:dx< -64?-.6:0,false);phase="wait_for_reachable_target";
            attack_frames_=0;attack_airborne_=false;
        } else if(ground && num(c,"obstacle_distance_pixels",999)<24) {
            proposal["phase"]="needs_reachable_approach";return proposal;
        } else
        if(ground) {
            if(attack_airborne_) {attack_frames_=0;attack_airborne_=false;}
            const auto prior=s.value("recent_control",J::object()).value("action",J(nullptr));
            if(attack_frames_>0 && attack_frames_<4) {
                ++attack_frames_;action=steer(std::clamp(dx*.03+evx,-1.3,1.3),true);phase="latch_takeoff";
            } else if(prior.is_string() && jumping(prior.get<std::string>())) {
                action=steer(0,false);phase="release_before_retry";attack_frames_=0;
            } else {
                const double closing=std::max(0.,(dx>=0?1:-1)*(vx-evx));
                const double takeoff=std::clamp(26.+closing*9.,30.,62.);
                if(std::abs(dx)<=takeoff) {
                    attack_frames_=1;action=steer(std::clamp(dx*.04+evx,-1.3,1.3),true);phase="launch_stomp";
                } else {
                    // Approach at walking speed and brake BEFORE jumping;
                    // holding right at running speed was skipping targets.
                    action=steer(dx>0?.8:-.8,false);phase="approach_target";
                }
            }
        } else {
            attack_airborne_=true;++attack_frames_;
            const double height=top-feet,vy=num(p,"vy");
            const double gravity=estimate(power(p)+":gravity_release",.75,.4,1.);
            const double eta=std::clamp((vy+std::sqrt(std::max(0.,vy*vy+2*gravity*height)))/gravity,2.,32.);
            const bool hold=attack_frames_<=8 && vy>0;
            const double desired=std::clamp(dx/eta+evx,-1.75,1.75);
            action=steer(desired,hold);phase="track_intercept";
            proposal["intercept_eta_frames"]=eta;proposal["desired_vx"]=desired;
        }
        proposal["apply"]=true;proposal["phase"]=phase;proposal["action"]=action;
        proposal["distance_px"]=dx;proposal["attack_frames"]=attack_frames_;
        return proposal;
    }
    J plans(const J& s,const J& task,const J& reach,const J& baseline,const J& intent) const {
        J plans=J::array();
        if(!yes(reach,"available")) return {{"candidates",plans},{"reason","insufficient_geometry"}};
        const auto p=s.at("player"),target=task.at("target");
        const bool nearby_task=task.at("status")=="active" && target.is_object() && std::abs(num(target,"x")-num(p,"x"))<128;
        const bool platform_route=platform_arena(s);
        const J landing_target=!yes(p,"grounded") && flight_.value("target",J(nullptr)).is_object()?
            flight_.at("target"):next_platform(s);
        if(!nearby_task && !platform_route)
            return {{"candidates",plans},{"reason","no_local_task_or_platform_requiring_rollout"}};
        const auto tiles=geometry(reach.at("tiles"));
        const auto moving_surfaces=s.value("collision",J::object()).value("moving_surfaces",J::array());
        const std::string base=baseline.value("action",std::string("noop"));
        struct Spec {std::string move;int hold,delay;bool baseline=false;int brake_after=99;};
        std::vector<Spec> specs{{base,0,0,true}};
        for(const auto* move:{"right_run","right","left","noop"}) {
            specs.push_back({move,0,0});
            for(int hold:{6,16,28}) for(int delay:{0,6}) specs.push_back({move,hold,delay});
        }
        if(platform_route) for(int hold:{16,28}) for(int brake:{12,20,28})
            specs.push_back({"right_run",hold,0,false,brake});
        const auto observed_input=s.value("recent_control",J::object()).value("action",J(nullptr));
        const std::string latched_input=observed_input.is_string()?observed_input.get<std::string>():"noop";
        const bool prior_a=jumping(preceding_input_);
        Body start{num(p,"x"),num(p,"feet_y"),num(p,"physics_vx",num(p,"vx")),-num(p,"vy"),yes(p,"grounded"),prior_a};
        start.cut=jump_cut_;start.air_cap=std::max(1.75,std::abs(start.vx));
        const auto mode=s.value("strategy",J::object()).value("mode",std::string("speedrun"));
        const bool hunt=(mode=="hunter" || mode=="score_attack") && task.at("status")=="active" && target.is_object();
        int id=0;
        for(const auto& spec:specs) {
            Body b=start;bool unknown=false,collision=false,stomp=false,ceiling=false,landed=false;
            int first_collision=49,landing_frame=0;double landing_x=0,landing_y=0;
            J path=J::array(),sequence=J::array(); std::set<int> simulated_kills;
            std::string latched=latched_input;
            double target_distance=target.is_object()?std::abs(num(target,"x")-b.x):0;
            for(int f=0;f<48;++f) {
                std::string action=spec.move;
                if(f>=spec.brake_after) action="left";
                if(!spec.baseline && f>=spec.delay && f<spec.delay+spec.hold)
                    action=action=="noop"?"jump":action+"_jump";
                if(spec.baseline && f>=16 && jumping(action)) action=action=="jump"?"noop":action.substr(0,action.size()-5);
                if(sequence.empty() || sequence.back().at("action")!=action) sequence.push_back({{"action",action},{"frames",1}});
                else sequence.back()["frames"]=sequence.back().at("frames").get<int>()+1;
                const auto old=b;auto m=advance(b,latched,tiles,power(p));b=m.body;latched=action;
                // Large platforms are RAM enemy-slot objects rather than
                // nametable tiles. Project only their observed short-horizon
                // motion and allow top contact, never side-wall collision.
                for(const auto& surface:moving_surfaces) {
                    if(!surface.is_object())continue;
                    const double shift=num(surface,"vx")*(f+1);
                    const double top=num(surface,"y")+num(surface,"vy")*(f+1);
                    const double left=num(surface,"left_x")+shift;
                    const double right=num(surface,"right_x")+shift;
                    if(b.x+10<=left || b.x+2>=right)continue;
                    if(b.vy>=0 && old.y<=top+2 && b.y>=top) {
                        b.y=top;b.vy=0;m.landed=!old.ground;b.ground=true;m.unknown=false;
                    } else if(old.ground && std::abs(old.y-top)<4) {
                        b.y=top;b.vy=0;b.ground=true;m.unknown=false;
                    }
                }
                unknown|=m.unknown;ceiling|=m.ceiling;landed|=m.landed;
                if(m.landed) {landing_frame=f+1;landing_x=b.x;landing_y=b.y;}
                for(const auto& [tid,t]:tracks_) if(t.visible && t.status!="confirmed_eliminated" && !simulated_kills.count(tid)) {
                    double left=t.x,right=t.x+16,top=t.y+8,bottom=top+16;
                    // Adjacent RAM slots with the same non-stompable kind and
                    // vertically offset sprites are one larger body, not two
                    // independently narrow enemies. This is observed shape,
                    // not a species-specific route or level coordinate.
                    if(!t.stompable) for(const auto& [oid,other]:tracks_)
                        if(oid!=tid && other.visible && other.kind==t.kind &&
                           std::abs(std::abs(other.x-t.x)-16)<=2 &&
                           std::abs(std::abs(other.y-t.y)-8)<=3) {
                            left=std::min(left,other.x);right=std::max(right,other.x+16);
                            top=std::min(t.y,other.y);
                            bottom=std::max(t.y,other.y)+24;
                        }
                    const double horizon=std::min(f+1,12);
                    left+=(t.velocity?t.vx:0)*(f+1);
                    right+=(t.velocity?t.vx:0)*(f+1);
                    top+=(t.velocity?t.vy:0)*horizon;
                    bottom+=(t.velocity?t.vy:0)*horizon;
                    if(b.x+12>left && b.x<right && b.y>top &&
                       b.y-(power(p)=="small"?16:32)<bottom) {
                        if(t.stompable && old.y<=top+5 && b.vy>=0) {
                            simulated_kills.insert(tid);stomp|=target.is_object() && num(target,"id",-1)==tid;
                            b.vy=-4;b.ground=false;
                        } else {collision=true;first_collision=std::min(first_collision,f+1);}
                    }
                }
                if(target.is_object()) target_distance=std::abs(num(target,"x")+num(target,"vx")*(f+1)-b.x);
                if(f%6==0 || f==47) path.push_back({{"frame",f+1},{"x",b.x},{"feet_y",b.y}});
                if(b.y>260) {collision=true;first_collision=std::min(first_collision,f+1);break;}
                // A narrow support is a subgoal, not a runway. Continuing the
                // same command after landing rejected otherwise safe jumps.
                if(platform_route && m.landed && !hunt) break;
            }
            const auto first=sequence.front().at("action").get<std::string>();
            const std::string skill=hunt?"stomp":start.ground?"traverse":"land";
            const auto signature=skill+":"+spec.move+":"+std::to_string(spec.hold)+":"+std::to_string(spec.delay)+
                (spec.brake_after<99?":brake:"+std::to_string(spec.brake_after):"");
            const int failed=contextual_failure_count(s,signature,landing_target);
            const double error=prediction_error_.n?prediction_error_.mean:2.;
            const double uncertainty=std::min(.25,error*.025)+(samples_<12?.12:0);
            bool target_landed=false;
            if(landed && landing_target.is_object()) {
                const bool moving_target=landing_target.value("source",std::string())==
                    "observed_ram_moving_support";
                const double width=num(landing_target,"right_x")-num(landing_target,"left_x");
                const double margin=moving_target?std::min(8.,width/4.):std::min(20.,width/2.);
                const double travel=moving_target?landing_frame:0;
                target_landed=std::abs(landing_y-num(landing_target,"y")-num(landing_target,"vy")*travel)<3 &&
                    landing_x+6>=num(landing_target,"left_x")+num(landing_target,"vx")*travel+margin &&
                    landing_x+6<=num(landing_target,"right_x")+num(landing_target,"vx")*travel-margin;
            }
            J continuation={{"status","not_applicable"},{"feasible",nullptr}};
            bool continuation_failed=false;
            if(platform_route && !hunt && landed) {
                const auto support=support_at(s,landing_x,landing_y),next=following_support(s,support);
                const double reserve=std::max(12.,std::max(0.,b.vx)*3.+4.);
                const double margin=support.is_object()?num(support,"right_x")-(landing_x+8):0;
                continuation={{"status",next.is_object()?"evaluated":"next_support_unknown"},
                    {"support",support},{"next_support",next},{"edge_margin_px",margin},
                    {"required_input_contact_reserve_px",reserve},{"feasible",nullptr},
                    {"semantics","approximate second transfer; not calibrated success probability"}};
                if(next.is_object()) {
                    bool feasible=false;int attempts=0;
                    // Releasing the previous jump and the one-frame input latch
                    // both consume support. A mere contact at the trailing edge
                    // is not a usable staging point for the following skill.
                    if(margin>=reserve) for(int hold:{16,28}) for(const auto* move:{"right","right_run"}) {
                        Body second=b;second.a=jumping(latched);second.cut=false;
                        std::string pending="noop";++attempts;
                        for(int t=0;t<48;++t) {
                            auto m=advance(second,pending,tiles,power(p));second=m.body;
                            pending=std::string(move)+(t>=1 && t<hold?"_jump":"");
                            if(m.unknown || m.ceiling || second.y>260) break;
                            if(m.landed && std::abs(second.y-num(next,"y"))<2 &&
                                second.x+6>=num(next,"left_x")+8 && second.x+6<=num(next,"right_x")-12) {
                                feasible=true;break;
                            }
                        }
                    }
                    continuation["feasible"]=feasible;continuation["evaluated_sequences"]=attempts;
                    continuation["status"]=feasible?"second_transfer_predicted":margin<reserve?"insufficient_takeoff_reserve":"no_second_transfer_predicted";
                    continuation_failed=!feasible;
                }
            }
            // A release near the landing edge is not robust to prediction error.
            const bool unsafe_cut=platform_route && !hunt && !start.ground && num(p,"vy")>0 &&
                jumping(latched_input) && !jumping(first) && landing_target.is_object() && !target_landed;
            const double risk=std::min(1.,(collision||unsafe_cut?1.:0.)+(unknown?.55:0.)+(!b.ground?.25:0.)+uncertainty+std::min(.15,failed*.03));
            double utility=(b.x-start.x)*.2-(ceiling?10:0)+(landed?8:0)-risk*100;
            // Unknown future geometry is not a veto. Observed infeasible
            // continuation is an explicit cost; emergency survival may still
            // require landing here rather than refusing every candidate.
            if(continuation_failed) utility-=160;
            else if(continuation.value("feasible",J(nullptr))==true) utility+=50;
            if(platform_route && !hunt && landed && landing_x>start.x+8) utility+=60;
            if(platform_route && !hunt && target_landed) utility+=100;
            if(hunt) utility= (stomp?200.:0.)-target_distance*.7+(landed?8:0)-risk*140;
            if(mode=="collector" || mode=="score_attack") {
                const auto d=s.value("detectors",J::object());
                if(num(d,"powerups_visible")>0) utility-=std::abs(start.x+num(d,"powerup_distance")-b.x)*.15;
            }
            utility-=20.*failed;
            const double objective_utility=utility;
            const auto goal=intent.value("active_goal",std::string());
            double model_bonus=0;
            if(yes(intent,"accepted")) {
                if(goal=="stomp_enemy" && stomp) model_bonus=30;
                if(goal=="finish_fast") model_bonus=(b.x-start.x)*.05;
                if(goal=="collect_powerup" && num(s.value("detectors",J::object()),"powerups_visible")>0)
                    model_bonus=-std::abs(start.x+num(s.at("detectors"),"powerup_distance")-b.x)*.2;
            }
            utility+=model_bonus;
            plans.push_back({{"id",id++},{"action",first},{"sequence",sequence},{"path",path},{"utility",utility},
                {"risk_cost",risk},{"unknown_space",unknown},{"predicted_collision",collision},{"predicted_stomp",stomp},
                {"ceiling_contact",ceiling},{"supported_end",b.ground},{"baseline",spec.baseline},{"skill",skill},
                {"first_collision_frame",first_collision},
                {"landing_frame",landing_frame},{"landing_x",landing_x},{"landing_y",landing_y},
                {"target_landed",target_landed},{"unsafe_jump_cut",unsafe_cut},{"landing_target",landing_target},
                {"continuation",continuation},{"continuation_failed",continuation_failed},
                {"signature",signature},
                {"objective_utility",objective_utility},{"model_bonus",model_bonus},
                {"uncertainty_cost",uncertainty},{"failure_penalty",failed},{"target_distance",target_distance}});
        }
        return {{"candidates",plans},{"horizon_frames",48},{"execution_frames",1},
            {"method","bounded action-sequence rollout with observed second-support continuation; replan every observation"}};
    }
    J risk(const J& s,const J& plans,const J& baseline,const J& task) const {
        const auto ep=s.value("episode",J::object()),session=s.value("session",J::object());
        const int lives=num(session,"lives_remaining",num(ep,"lives_remaining",3));
        const double budget=lives<=1?.42:lives==2?.52:.65;
        J best=nullptr,without_model=nullptr,emergency=nullptr;int eligible=0;
        for(const auto& plan:plans.at("candidates")) {
            if(emergency.is_null() || num(plan,"first_collision_frame")>num(emergency,"first_collision_frame") ||
                (num(plan,"first_collision_frame")==num(emergency,"first_collision_frame") && num(plan,"utility")>num(emergency,"utility"))) emergency=plan;
        }
        for(const auto& plan:plans.at("candidates")) if(!yes(plan,"predicted_collision") && num(plan,"risk_cost")<=budget) {
            ++eligible;if(best.is_null() || num(plan,"utility")>num(best,"utility")) best=plan;
            if(without_model.is_null() || num(plan,"objective_utility")>num(without_model,"objective_utility")) without_model=plan;
        }
        const bool terminal=yes(ep,"dead") || yes(ep,"stage_clear") || yes(session,"game_over") || yes(session,"game_complete") || yes(session,"life_lost") || yes(session,"stage_transition");
        const auto reason=baseline.value("reason",std::string());
        const bool macro=reason.find("pipe")!=std::string::npos || reason.find("passage")!=std::string::npos || reason.find("recovery_")==0;
        const auto p=s.value("player",J::object()),col=s.value("collision",J::object());
        const bool platforms=platform_arena(s);
        const auto target=task.at("target");
        const bool near_target=target.is_object() && std::abs(num(target,"x")-num(p,"x"))<128;
        const bool clear_arena=num(col,"gap_distance_pixels",999)>96 && num(col,"obstacle_distance_pixels",999)>64;
        const bool hunting=task.at("status")=="active" && near_target && clear_arena;
        // The learned model is approximate: it must not cancel a committed
        // gap/wall maneuver just because a short rollout prefers a slower hop.
        const auto trajectory=s.value("trajectory",J::object());
        const bool platform_landing=platforms && !best.is_null() && yes(best,"target_landed") &&
            num(best,"landing_frame")>0 &&
            num(best,"landing_x")>num(p,"x")+8 && !yes(best,"predicted_collision") && !yes(best,"unknown_space");
        const bool committed=(!platform_landing && (reason=="gap" || reason=="obstacle" || reason=="enemy_below_edge")) ||
            (!platforms && (yes(trajectory,"crossing_gap") || yes(trajectory,"crossing_obstacle")));
        const bool needed=(hunting || platforms) && !committed;
        const bool emergency_used=best.is_null() && !emergency.is_null();
        if(emergency_used) best=emergency;
        const bool enabled=s.value("planner_enabled",true);
        const bool takeoff_latch=yes(p,"grounded") && !flight_.empty() && !yes(flight_,"airborne_observed") &&
            frame_-num(flight_,"start_frame")<4;
        const bool objective_evidence=!hunting || (!best.is_null() && yes(best,"predicted_stomp"));
        const bool choose=enabled && !terminal && !macro && !takeoff_latch && needed && objective_evidence && !best.is_null() &&
            (!emergency_used || (hunting && target.is_object() && std::abs(num(target,"x")-num(p,"x"))<32));
        return {{"budget",budget},{"lives",lives},{"eligible_candidates",eligible},{"selected",best},
            {"apply",choose},{"emergency",emergency_used},
            {"model_changed_selected_action",!best.is_null() && !without_model.is_null() && best.at("action")!=without_model.at("action")},
            {"reason",!enabled?"explicit_baseline_ablation":terminal?"lifecycle_guard":macro?"navigation_macro":takeoff_latch?"takeoff_input_latch":committed?"committed_geometry_guard":!needed?"baseline_outside_verified_planning_arena":!objective_evidence?"no_predicted_task_completion":best.is_null() || (emergency_used && !choose)?"no_plan_within_risk_budget":emergency_used?"no_safe_plan_maximize_time_to_collision":"best_feasible_rollout"},
            {"risk_semantics","heuristic cost, NOT probability of death"},{"failures_by_skill",failures_},
            {"failures_by_skill_semantics","diagnostic totals only; NOT applied as cross-context penalties"}};
    }
    J execute(const J& s,const J& baseline,const J& risk,const J& task,const J& combat) {
        namespace skills=jevt::skills;
        const auto p=s.value("player",J::object()),col=s.value("collision",J::object());
        const auto lifecycle=s.value("session",J::object()),episode=s.value("episode",J::object());
        const bool terminal=yes(episode,"dead") || yes(episode,"stage_clear") || yes(lifecycle,"life_lost") ||
            yes(lifecycle,"stage_transition") || yes(lifecycle,"game_over") || yes(lifecycle,"game_complete");
        if(terminal) paired_hazard_phase_=0;
        const auto base_reason=baseline.value("reason",std::string());
        const bool navigation=base_reason.find("pipe")!=std::string::npos || base_reason.find("passage")!=std::string::npos ||
            base_reason.find("recovery_")==0;
        const skills::snapshot stamp{std::to_string(run_),static_cast<std::uint64_t>(std::max(0,frame_)),
            static_cast<std::uint64_t>(std::max(0,life_)),static_cast<std::uint64_t>(std::max(0,frame_))};
        std::vector<skills::proposal<J>> proposals;
        auto add=[&](std::string id,std::string skill,J payload,bool allowed,int priority,std::string detail) {
            payload["skill_signature"]=id=="rollout"?risk.at("selected").value("signature",skill):skill+":"+payload.at("action").get<std::string>();
            proposals.push_back({id,skill,payload.value("source",std::string("jevtpp_reactive_controller")),payload,stamp,0,priority,0,false,
                {{"execution_contract",allowed?skills::truth::true_value:skills::truth::false_value,detail}}});
        };
        add("reactive","baseline",baseline,true,terminal?1000:navigation?900:100,
            terminal?"lifecycle guard":navigation?"committed navigation":"reactive fallback; not a safety guarantee");
        if(base_reason=="intercept_observed_moving_support")
            add("moving_intercept","observed_moving_support",baseline,!terminal,760,
                "live RAM support position and velocity; update interception every frame");
        if(base_reason=="runup_on_observed_moving_support")
            add("moving_runup","observed_moving_support_runup",baseline,!terminal,740,
                "preserve runway and accelerate before next moving-support transfer");
        // Two adjacent non-stompable RAM slots offset vertically describe a
        // larger oscillating body. Stage a crossing from live entity motion:
        // wait while it is above the current lane, accelerate as it descends,
        // then commit a single jump. This uses no saved map or level x value.
        // Continue tracking after one half crosses behind Mario: the
        // upcoming-only list drops it before the body has been cleared.
        const auto nearby=s.value("hazard",J::object()).value("nearby_enemies",J::array());
        J paired=J(nullptr);double paired_top=0,paired_dx=999;
        for(std::size_t i=0;i<nearby.size();++i) for(std::size_t j=i+1;j<nearby.size();++j) {
            const auto& a=nearby[i];const auto& b=nearby[j];
            if(yes(a,"stompable") || yes(b,"stompable") ||
               num(a,"kind_id",-1)!=num(b,"kind_id",-2) ||
               std::abs(std::abs(num(a,"x")-num(b,"x"))-16)>2 ||
               std::abs(std::abs(num(a,"screen_y")-num(b,"screen_y"))-8)>3)continue;
            const double dx=std::min(num(a,"relative_x_pixels"),num(b,"relative_x_pixels"));
            if(dx<paired_dx) {paired=a;paired_dx=dx;
                paired_top=std::min(num(a,"screen_y"),num(b,"screen_y"));}
        }
        if(!terminal && !navigation && paired_hazard_phase_==0 && paired.is_object() &&
           yes(p,"grounded") && paired_dx>=35 && paired_dx<=52 &&
           paired_top<num(p,"screen_y")-16 &&
           num(col,"gap_distance_pixels",999)>96 &&
           num(col,"obstacle_distance_pixels",999)>64) {
            paired_hazard_phase_=1;paired_hazard_started_=frame_;paired_hazard_jump_=0;
        }
        if(paired_hazard_phase_ && (frame_-paired_hazard_started_>100 ||
           (paired_hazard_phase_<3 && !paired.is_object()))) paired_hazard_phase_=0;
        if(paired_hazard_phase_ && !terminal && !navigation) {
            std::string action="noop", phase="wait_for_observed_hazard_descent";
            if(paired_hazard_phase_==1 && paired_top>=num(p,"screen_y")-2)
                paired_hazard_phase_=2;
            if(paired_hazard_phase_==2) {
                action="right_run";phase="approach_observed_hazard_window";
                if(paired_dx<=25 && num(p,"physics_vx",num(p,"vx"))>=1.65)
                    paired_hazard_phase_=3;
            }
            if(paired_hazard_phase_==3) {
                action="right_run_jump";phase="cross_observed_paired_hazard";
                if(++paired_hazard_jump_>=32)paired_hazard_phase_=4;
            } else if(paired_hazard_phase_==4) {
                action="right_run";phase="clear_observed_paired_hazard";
                if(!paired.is_object() || paired_dx<-32)paired_hazard_phase_=0;
            }
            J crossing=baseline;crossing["action"]=action;crossing["reason"]=phase;
            crossing["source"]="jevtpp_observed_hazard_skill";
            crossing["phase"]=phase;crossing["paired_hazard_phase"]=paired_hazard_phase_;
            crossing["strategy_applied"]=false;crossing["model_changed_action"]=false;
            crossing["model_effect"]="current_RAM_paired_body_and_phase";
            add("paired_hazard","observed_paired_hazard_crossing",crossing,true,750,
                "live two-slot entity and verified floor; bounded wait approach jump");
        }
        // Replanning each frame must not repeatedly execute just the first
        // action of a multi-step landing forecast. Commit only a forecast that
        // reached an observed support and verified a second transfer; abort
        // on contact, changed geometry, new immediate threat or divergence.
        if(!committed_landing_.empty()) {
            const int elapsed=frame_-committed_landing_frame_;
            const auto target=committed_landing_.value("landing_target",J(nullptr));
            bool support_observed=false;
            if(target.is_object())
                for(const auto& surface:col.value("landing_surfaces",J::array()))
                    support_observed|=std::abs(num(surface,"y")-num(target,"y"))<2 &&
                        num(surface,"left_x")<num(target,"right_x") &&
                        num(surface,"right_x")>num(target,"left_x");
            bool trajectory_consistent=true;
            for(const auto& point:committed_landing_.value("path",J::array())) {
                if(std::abs(num(point,"frame")-elapsed)>2)continue;
                trajectory_consistent=std::abs(num(point,"x")-num(p,"x"))<28 &&
                    std::abs(num(point,"feet_y")-num(p,"feet_y"))<24;
                break;
            }
            const bool valid=!terminal && !navigation && !yes(p,"grounded") &&
                elapsed>0 && elapsed<=std::min(48,static_cast<int>(num(committed_landing_,"landing_frame",48))+3) &&
                support_observed && trajectory_consistent &&
                num(s.value("hazard",J::object()),"enemy_distance",999)>32;
            if(valid) {
                std::string step="noop";int remaining=elapsed;
                for(const auto& segment:committed_landing_.value("sequence",J::array())) {
                    const int duration=segment.value("frames",0);
                    if(remaining<duration) {step=segment.value("action",std::string("noop"));break;}
                    remaining-=duration;
                }
                J continued=baseline;continued["action"]=step;
                continued["reason"]="committed_observed_landing_sequence";
                continued["source"]="jevtpp_online_planner";
                continued["plan_phase_frame"]=elapsed;
                continued["plan_signature"]=committed_landing_.value("signature",std::string());
                continued["strategy_applied"]=false;continued["model_changed_action"]=false;
                continued["model_effect"]="previously_verified_continuation_not_new_model_output";
                add("plan_commit","observed_landing_sequence",continued,true,650,
                    "previous forecast reached observed support with feasible next transfer");
            } else {committed_landing_=J::object();committed_landing_frame_=-1;}
        }
        // A higher adjacent support may require horizontal speed before A is
        // pressed. Jumping immediately after landing clamps the airborne
        // speed to the slow takeoff velocity and can make the next ledge
        // unreachable even while the vertical jump itself is high enough.
        const auto standing_support=yes(p,"grounded")?
            support_at(s,num(p,"x"),num(p,"feet_y")):J(nullptr);
        const auto next_support=following_support(s,standing_support);
        const double support_width=standing_support.is_object()?
            num(standing_support,"right_x")-num(standing_support,"left_x"):0;
        const double support_margin=standing_support.is_object()?
            num(standing_support,"right_x")-num(p,"x")-12:0;
        const double rise=standing_support.is_object() && next_support.is_object()?
            num(standing_support,"y")-num(next_support,"y"):0;
        const double transfer_gap=standing_support.is_object() && next_support.is_object()?
            num(next_support,"left_x")-num(standing_support,"right_x"):0;
        const bool staged_ascent=!terminal && !navigation && yes(p,"grounded") &&
            standing_support.is_object() && next_support.is_object() &&
            rise>=48 && rise<=80 && support_width>=48 &&
            num(next_support,"left_x")<=num(standing_support,"right_x")+80 &&
            num(next_support,"left_x")>=num(standing_support,"right_x") &&
            (transfer_gap<=32 || support_width<=48) &&
            support_margin>=18 && support_margin<=112 &&
            num(s.value("hazard",J::object()),"enemy_distance",999)>40;
        if(staged_ascent) {
            const bool tall_narrow_step=support_width<=48 && rise>=64 && transfer_gap<=16;
            const double takeoff_x=tall_narrow_step?
                num(standing_support,"right_x")-39:
                transfer_gap>16?
                num(standing_support,"right_x")-30:
                num(standing_support,"left_x")+std::min(20.,(support_width-24.)/2.);
            const bool ready=num(p,"x")>=takeoff_x &&
                num(p,"physics_vx",num(p,"vx"))>=(tall_narrow_step?1.35:2.1);
            J staging=baseline;staging["action"]=ready?"right_run_jump":"right_run";
            staging["reason"]=ready?"observed_support_transfer_takeoff":
                "observed_support_takeoff_staging";
            staging["source"]="jevtpp_skill_contract";
            staging["support_margin_px"]=support_margin;
            staging["next_support_rise_px"]=rise;
            staging["strategy_applied"]=false;staging["model_changed_action"]=false;
            staging["model_effect"]="current_RAM_support_and_takeoff_speed";
            add("platform_stage","observed_support_transfer",staging,true,720,
                ready?"observed takeoff position and horizontal speed ready":
                    "accelerate on observed support before higher transfer");
        }
        J planner=baseline;
        if(risk.value("selected",J(nullptr)).is_object()) {
            const auto selected=risk.at("selected"); const auto action=selected.at("action").get<std::string>();
            planner["action"]=action;planner["reason"]="receding_horizon_"+selected.at("skill").get<std::string>();
            planner["source"]="jevtpp_online_planner";
            planner["strategy_applied"]=risk.at("model_changed_selected_action");
            planner["model_changed_action"]=risk.at("model_changed_selected_action");
            planner["model_effect"]=yes(risk,"model_changed_selected_action")?"feasible_plan_ranking_changed_action":"same_action_without_model_bonus";
            planner["mode_changed_action"]=action!=baseline.at("action").get<std::string>();
            planner["mode_effect"]=task.at("objective");
            // Model rank is advisory in hunter mode until paired full-game
            // evidence supports promoting it over the target skill. A direct
            // priority promotion regressed both tested CUDA backends.
            add("rollout",selected.at("skill"),planner,yes(risk,"apply") && !terminal,
                400,risk.at("reason"));
        }
        J attack=baseline;attack["action"]=combat.value("action",J(nullptr)).is_string()?combat.at("action"):baseline.at("action");
        attack["reason"]="hunter_"+combat.at("phase").get<std::string>();
        attack["source"]="jevtpp_target_skill";attack["mode_effect"]="persistent_target_execution";
        attack["mode_changed_action"]=attack.at("action")!=baseline.at("action");
        attack["strategy_applied"]=false;attack["model_changed_action"]=false;attack["model_effect"]="user_objective_target_skill";
        add("target","stomp",attack,yes(combat,"apply") && !terminal && !navigation,
            combat.value("phase",std::string())=="release_before_retry"?800:450,combat.at("phase"));
        // A repeated, RAM-verified star contact outcome can offer a narrow
        // skill. This never relaxes gap/wall guards and never treats a model
        // statement as evidence. Recheck the star timer and enemy every frame.
        const auto affordance=s.value("knowledge_affordances",J::object());
        const auto hazard=s.value("hazard",J::object());
        const int star_timer=static_cast<int>(num(p,"star_timer_ram_0x079f",0));
        const auto mode=s.value("strategy",J::object()).value("mode",std::string());
        bool star_target=false;
        // SMB loads this counter with 0x23, not a frame count. Eight units
        // stay above the end-of-effect flashing band; do not interpret 8 as
        // eight remaining simulation frames.
        if(yes(p,"star_active") && star_timer>=8 && yes(p,"grounded") &&
           (mode=="hunter" || mode=="score_attack") &&
           col.value("available",false) && num(col,"gap_distance_pixels",999)>96 &&
           num(col,"obstacle_distance_pixels",999)>64 && !terminal && !navigation &&
           affordance.value("schema",std::string())=="jevt.mario_affordance.v1") {
            const auto kinds=affordance.value("star_contact_kinds",J::array());
            for(const auto& enemy:hazard.value("upcoming_enemies",J::array())) {
                if(!enemy.is_object() || enemy.value("category",std::string())!="hostile") continue;
                const double dx=num(enemy,"relative_x_pixels",999);
                // Begin before the ordinary enemy-jump trigger (~52 px), or
                // the committed jump would already own this encounter.
                if(dx<8 || dx>64 || std::abs(num(enemy,"relative_y_pixels",999))>32)continue;
                if(std::find(kinds.begin(),kinds.end(),enemy.value("kind_id",-1))!=kinds.end()) {
                    star_target=true;break;
                }
            }
        }
        J star=baseline;star["action"]="right_run";star["reason"]="empirical_star_contact";
        star["source"]="jevtpp_verified_knowledge_skill";
        star["mode_effect"]="star_contact_with_observed_enemy_type";
        star["model_effect"]="repeated_RAM_outcome_evidence_not_model_claim";
        star["strategy_applied"]=false;star["model_changed_action"]=false;
        add("star_contact","observed_star_contact",star,star_target,600,
            "repeated star-contact eliminations; current star timer and safe terrain verified");
        // A freshly verified takeoff is a skill phase, not an independent
        // invitation to release A next frame. Preserve its upward input until
        // the observed enemy collision volume has been cleared. This is an
        // explicit bounded proposal, not a mutation of the selected rollout.
        const auto target=task.value("target",J(nullptr));
        const auto actual_input=s.value("recent_control",J::object()).value("action",J(nullptr));
        bool ceiling_near=false;
        for(const auto& column:col.value("planning_columns",col.value("columns",J::array())))
            if(std::abs(num(column,"x")-num(p,"x"))<32)
                for(const auto& y:column.value("solid_y",J::array()))
                    ceiling_near|=y.get<double>()+16<=num(p,"feet_y")-(power(p)=="small"?16:32)+8 &&
                        y.get<double>()+16>num(p,"feet_y")-(power(p)=="small"?16:32)-48;
        if(target.is_object() && actual_input.is_string() && jumping(actual_input.get<std::string>()) &&
            yes(flight_,"takeoff_confirmed") && !yes(p,"grounded") && num(p,"vy")>0 &&
            frame_-num(flight_,"start_frame")<10 && std::abs(num(target,"x")-num(p,"x"))<64 &&
            num(p,"feet_y")>num(target,"screen_y") && !ceiling_near && !terminal && !navigation) {
            J committed=baseline;committed["action"]=actual_input;
            committed["reason"]="verified_takeoff_clearance";committed["source"]="jevtpp_skill_contract";
            committed["strategy_applied"]=false;committed["model_changed_action"]=false;
            committed["model_effect"]="observed_takeoff_clearance_invariant";
            add("takeoff_commitment","stomp_takeoff",committed,true,700,"rising below observed target clearance; no nearby ceiling");
        }
        // A planned ascent is not complete merely because the next one-frame
        // rollout prefers releasing A. NES jump height depends on continued
        // input; releasing below the observed landing ledge makes that ledge
        // unreachable. The commitment is bounded by the current flight and
        // current RAM geometry, and ends as soon as clearance is observed.
        const auto flight_target=flight_.value("target",J(nullptr));
        bool target_still_observed=false;
        if(flight_target.is_object())
            for(const auto& surface:col.value("landing_surfaces",J::array()))
                target_still_observed|=std::abs(num(surface,"y")-num(flight_target,"y"))<2 &&
                    num(surface,"left_x")<num(flight_target,"right_x") &&
                    num(surface,"right_x")>num(flight_target,"left_x");
        const bool climbing_to_platform=flight_target.is_object() && target_still_observed &&
            num(flight_target,"y")<num(flight_,"start_feet_y")-24 &&
            num(flight_target,"left_x")>num(flight_,"start_x")+12 &&
            num(p,"feet_y")>num(flight_target,"y")-8 &&
            num(p,"x")<num(flight_target,"right_x")-16;
        if(climbing_to_platform && actual_input.is_string() && jumping(actual_input.get<std::string>()) &&
            yes(flight_,"takeoff_confirmed") && !yes(p,"grounded") && num(p,"vy")>0 &&
            !jump_cut_ && frame_-num(flight_,"start_frame")<30 &&
            !ceiling_near && !terminal && !navigation) {
            J committed=baseline;committed["action"]=actual_input;
            committed["reason"]="observed_platform_ascent_commitment";
            committed["source"]="jevtpp_skill_contract";
            committed["phase"]="ascent_to_observed_support";
            committed["target_support"]={{"left_x",flight_target.value("left_x",0)},
                {"right_x",flight_target.value("right_x",0)},{"y",flight_target.value("y",0)}};
            committed["strategy_applied"]=false;committed["model_changed_action"]=false;
            committed["model_effect"]="current_RAM_support_and_confirmed_takeoff";
            add("platform_hold","observed_support_ascent",committed,true,700,
                "confirmed ascent below observed support; jump release would reduce height");
        }
        // Adaptation is a visible pre-arbitration constraint on proposals, never
        // a hidden actuator write after a different plan has been selected.
        const auto flight_context=flight_.value("context",std::string());
        const bool hold_constraint=s.value("planner_enabled",true) && !terminal && !navigation &&
            !yes(p,"grounded") && num(p,"vy")>0 && !jump_cut_ && yes(flight_,"takeoff_confirmed") &&
            minimum_hold_.contains(flight_context) && num(flight_,"held_frames")<minimum_hold_.at(flight_context);
        for(auto& proposal:proposals) {
            auto action=proposal.action.at("action").get<std::string>();
            // Do not modify a rollout's evaluated input sequence or a combat
            // intercept. Adaptation may constrain only an otherwise unplanned
            // reactive action; ceiling proximity conservatively vetoes it.
            bool overhead=false;
            for(const auto& column:col.value("planning_columns",col.value("columns",J::array())))
                if(std::abs(num(column,"x")-num(p,"x"))<32)
                    for(const auto& y:column.value("solid_y",J::array()))
                        overhead|=y.get<double>()+16<=num(p,"feet_y")-(power(p)=="small"?16:32)+8 &&
                            y.get<double>()+16>num(p,"feet_y")-(power(p)=="small"?16:32)-64;
            if(hold_constraint && proposal.id=="reactive" && !yes(risk,"apply") && !yes(combat,"apply") &&
                !overhead && !jumping(action)) {
                proposal.action["action"]=action=="noop"?"jump":action+"_jump";
                proposal.action["input_constraint"]="contextual_confirmed_takeoff_minimum_hold";
                proposal.action["strategy_applied"]=false;proposal.action["model_changed_action"]=false;
                proposal.action["model_effect"]="action_constrained_by_same_run_outcome_evidence";
            }
        }
        const auto decision=skills::arbitrate<J>(stamp,proposals);
        J result=decision.action.value_or(baseline),trace=J::array();
        for(const auto& item:decision.trace) trace.push_back({{"id",item.id},{"skill",item.skill_id},{"source",item.source},
            {"status",std::string(skills::status_name(item.status))},{"reason",item.reason}});
        result["arbitration"]={{"selected",decision.selected_id},{"proposals",trace},
            {"single_writer",true},{"post_selection_mutation",false},{"frame",frame_}};
        skill_=decision.selected_id=="target"?"stomp":decision.selected_id=="rollout"?risk.at("selected").value("skill",std::string("traverse")):"baseline";
        plan_key_=result.value("skill_signature",std::string("baseline:"+result.at("action").get<std::string>()));
        result["hunter_engagement"]=combat;
        result["planner_applied"]=decision.selected_id=="rollout";result["reactive_action"]=baseline.at("action");
        result["task_status"]=task.at("status");result["task_target"]=task.at("target");
        result["planner_reason"]=risk.at("reason");
        if(decision.selected_id=="rollout" && committed_landing_.empty() &&
            risk.value("selected",J(nullptr)).is_object()) {
            const auto& selected=risk.at("selected");
            if(!yes(p,"grounded") && yes(selected,"target_landed") &&
                selected.value("continuation",J::object()).value("feasible",J(nullptr))==true &&
                num(selected,"landing_frame")>1 && !yes(selected,"predicted_collision") &&
                num(selected,"risk_cost",1)<=num(risk,"budget",0)) {
                committed_landing_=selected;committed_landing_frame_=frame_;
            }
        }
        previous_executed_=result.at("action");
        if(yes(p,"grounded") && jumping(previous_executed_) && flight_.empty() &&
            !yes(s.value("episode",J::object()),"dead") && !yes(s.value("session",J::object()),"life_lost")) {
            const auto credited_target=decision.selected_id=="rollout"?
                risk.at("selected").value("landing_target",next_platform(s)):next_platform(s);
            flight_={{"signature",plan_key_},{"start_frame",frame_},{"start_x",num(p,"x")},
                {"start_feet_y",num(p,"feet_y")},{"start_vx",num(p,"physics_vx")},
                {"apex_feet_y",num(p,"feet_y")},{"airborne_observed",false},
                {"result","awaiting_landing"},{"target",credited_target},
                {"surfaces_at_takeoff",col.value("landing_surfaces",J::array())}};
            flight_["credit_key"]=failure_key(s,plan_key_,credited_target);
            flight_["context"]=flight_.at("credit_key");
        }
        if(!flight_.empty() && !yes(flight_,"airborne_observed") && frame_-num(flight_,"start_frame")>8) {
            flight_["result"]="takeoff_not_observed";flight_["termination"]="blocked_or_input_not_latched";
            last_flight_=flight_;flight_=J::object();
        }
        Body b{num(p,"x"),num(p,"feet_y"),num(p,"physics_vx",num(p,"vx")),-num(p,"vy"),yes(p,"grounded"),jumping(preceding_input_)};
        b.cut=jump_cut_;b.air_cap=std::max(1.75,std::abs(b.vx));
        const auto input=s.value("recent_control",J::object()).value("action",J(nullptr));
        const auto m=advance(b,input.is_string()?input.get<std::string>():"noop",geometry(col.value("planning_columns",col.value("columns",J::array()))),power(p));
        prediction_={{"x",m.body.x},{"feet_y",m.body.y}};
        return result;
    }
};
}
