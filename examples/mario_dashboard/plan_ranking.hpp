#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

// Mario adapter for a bounded candidate-ranking contract. Scores are preferences,
// not success probabilities. Every selected option is rechecked against live plans.
namespace mario::ranking {
using J=nlohmann::json;
inline bool finite_cost(const J& p,const char* key) {
    const auto it=p.find(key);return it!=p.end()&&it->is_number()&&std::isfinite(it->get<double>())&&it->get<double>()>=0;
}
inline std::string fingerprint(const J& value) {
    std::uint64_t hash=14695981039346656037ULL;
    for(unsigned char c:value.dump()){hash^=c;hash*=1099511628211ULL;}
    return std::to_string(hash);
}
inline J offer(const J& state,const J& compact,const J& plans,const J& task) {
    const auto player=state.value("player",J::object());
    const auto target=task.value("target",J(nullptr));
    J facts{{"level",state.value("level",J::object())},{"mode",compact.value("mode",std::string())},
            {"grounded",player.value("grounded",false)},{"power",player.value("powerup_status",std::string())},
            {"task_status",task.value("status",std::string())},{"target_id",target.is_object()?target.value("id",-1):-1},
            {"attack_path_blocked",compact.value("attack_path_blocked",false)}};
    std::vector<J> eligible;
    const bool hunter=task.value("objective",std::string())=="eliminate_observed_hostiles";
    for(const auto& plan:plans.value("candidates",J::array())) {
        if(plan.value("predicted_collision",true)||plan.value("unknown_space",true)||plan.value("unsafe_jump_cut",false)||
           plan.value("continuation_failed",false)||!finite_cost(plan,"risk_cost")||
           (hunter&&!plan.value("predicted_stomp",false))||
           !plan.contains("objective_utility")||!plan.at("objective_utility").is_number()||
           !std::isfinite(plan.at("objective_utility").get<double>()))continue;
        eligible.push_back(plan);
    }
    std::stable_sort(eligible.begin(),eligible.end(),[](const J& a,const J& b){return a.value("objective_utility",0.)>b.value("objective_utility",0.);});
    J candidates=J::array();std::set<std::string> seen;
    for(const auto& plan:eligible) {
        const auto id=plan.value("signature",std::string());
        if(id.empty()||!seen.insert(id).second)continue;
        J brief{{"id",id},{"slot",candidates.size()},{"action",plan.at("action")}};
        for(const auto* key:{"skill","sequence","risk_cost","uncertainty_cost","landing_target","landing_x","landing_y","landing_frame","predicted_stomp","supported_end","continuation","failure_penalty"})
            if(plan.contains(key))brief[key]=plan.at(key);
        candidates.push_back(brief);
        if(candidates.size()==6)break;
    }
    J targets=J::array(),candidate_targets=J::array();
    for(const auto& candidate:candidates) {
        auto target=candidate.value("landing_target",J(nullptr));
        J geometry=nullptr;
        if(target.is_object()) {
            geometry={{"left_x",target.value("left_x",J(nullptr))},
                      {"right_x",target.value("right_x",J(nullptr))},{"y",target.value("y",J(nullptr))}};
            targets.push_back(geometry);
        }
        // Keep ID -> target association: independently sorted sets would accept
        // a stale ranking when the same action IDs exchanged target platforms.
        candidate_targets.push_back({{"id",candidate.at("id")},{"target",geometry}});
    }
    std::sort(targets.begin(),targets.end());
    targets.erase(std::unique(targets.begin(),targets.end()),targets.end());
    facts["landing_targets"]=targets;
    std::sort(candidate_targets.begin(),candidate_targets.end(),[](const J& a,const J& b){return a.at("id")<b.at("id");});
    facts["candidate_targets"]=candidate_targets;
    facts["candidate_ids"]=J::array();
    for(const auto& c:candidates)facts["candidate_ids"].push_back(c.at("id"));
    std::sort(facts["candidate_ids"].begin(),facts["candidate_ids"].end());
    const auto context=fingerprint(facts);
    return {{"schema","jevt.plan_candidates.v1"},{"context_id",context},{"context_facts",facts},
            {"plan_candidates",candidates},{"observation",compact},
            {"limitations","Bounded observed rollout proposals; risk is a heuristic cost; absent slots must not be selected"}};
}
inline J validate(const J& input,const J& offered) {
    const auto cached=input.value("strategy",J::object());
    const int age=input.value("frame",0)-cached.value("source_frame",-10000);
    std::string reason="missing_plan_judgment";
    const auto scores=cached.value("plan_scores",J::object());
    const auto selected=cached.value("model_plan",std::string());
    bool accepted=false;
    if(cached.value("judgment_kind",std::string())=="plan") {
        if(!cached.value("ok",false))reason="model_error";
        else if(cached.value("epoch",-1)!=input.value("epoch",0)||age<0||age>45)reason="stale_epoch_or_age";
        else if(cached.value("mode",std::string())!=offered.at("observation").value("mode",std::string()))reason="objective_changed";
        else if(selected.empty())reason="model_abstained_or_selected_absent_slot";
        else {
            const bool exact=cached.value("plan_context",std::string())==offered.at("context_id").get<std::string>();
            const auto old=cached.value("plan_facts",J::object());
            const auto now=offered.at("context_facts");
            bool relevant=old.is_object()&&old.contains("candidate_targets")&&old.at("candidate_targets").is_array();
            if(!exact)for(const auto* key:{"level","mode","grounded","power","task_status","target_id","attack_path_blocked"})
                if(!old.contains(key)||old.at(key)!=now.at(key))relevant=false;
            J old_target=nullptr,current_target=nullptr;
            bool old_present=false,current_present=false;
            if(relevant)for(const auto& entry:old.at("candidate_targets"))if(entry.value("id",std::string())==selected) {
                old_target=entry.value("target",J(nullptr));old_present=true;
            }
            for(const auto& entry:now.at("candidate_targets"))if(entry.value("id",std::string())==selected) {
                current_target=entry.value("target",J(nullptr));current_present=true;
            }
            accepted=current_present&&(exact||(relevant&&old_present&&old_target==current_target));
            reason=accepted?(exact?"current_context_and_candidate_verified":"selected_plan_and_relevant_context_verified"):
                            exact?"candidate_no_longer_available":"relevant_context_changed";
            if(accepted) {
                if(!scores.is_object()||!scores.contains(selected))accepted=false;
                else for(const auto& [id,score]:scores.items())
                    if(!score.is_number()||!std::isfinite(score.get<double>())||score.get<double>()<0||score.get<double>()>1)accepted=false;
                    else {
                        const auto old_ids=old.value("candidate_ids",now.at("candidate_ids"));
                        bool originally_offered=false;
                        for(const auto& candidate_id:old_ids)if(candidate_id==id)originally_offered=true;
                        if(!originally_offered)accepted=false;
                    }
                if(!accepted)reason="invalid_plan_scores";
            }
        }
    }
    return {{"accepted",accepted},{"reason",reason},{"age_frames",age},{"selected",selected},
            {"source_context",cached.value("plan_context",std::string())},{"current_context",offered.at("context_id")},
            {"scores",accepted?scores:J::object()},{"semantics","preference among recomputed feasible plans; not win probability"}};
}
inline J apply(J risk,const J& plans,const J& gate,const J& task) {
    risk["plan_judgment"]=gate;
    risk["model_rank_applied"]=false;
    if(!gate.value("accepted",false)||!risk.value("apply",false)||risk.value("emergency",false)||!finite_cost(risk,"budget"))return risk;
    const auto scores=gate.at("scores");J best=nullptr;double high=-1;
    const bool hunt=task.value("objective",std::string())=="eliminate_observed_hostiles";
    for(const auto& p:plans.at("candidates")) {
        const auto id=p.value("signature",std::string());
        if(!scores.contains(id)||p.value("predicted_collision",true)||p.value("unknown_space",true)||
           p.value("continuation_failed",false)||!finite_cost(p,"risk_cost")||
           p.value("unsafe_jump_cut",false)||p.value("risk_cost",1.)>risk.value("budget",0.)||
           (hunt&&!p.value("predicted_stomp",false)))continue;
        const double score=scores.at(id).get<double>();
        if(score>high){high=score;best=p;}
    }
    if(best.is_null())return risk;
    const auto original=risk.value("selected",J(nullptr));
    risk["model_changed_selected_action"]=original.is_object()&&original.at("action")!=best.at("action");
    risk["model_free_selected"]=original;
    risk["selected"]=best;risk["model_rank_applied"]=true;
    risk["reason"]="selected_model_rank_within_current_risk_budget";
    return risk;
}
}
