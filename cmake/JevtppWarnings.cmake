function(jevtpp_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target} PRIVATE
      -Wall
      -Wextra
      -Wconversion
      -Wpedantic
      -Wshadow
    )
  endif()
endfunction()

