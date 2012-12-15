FILE(REMOVE_RECURSE
  "./CUDARuntime_generated_CUSPEngine.cu.o"
  "libCUDARuntime.pdb"
  "libCUDARuntime.a"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/CUDARuntime.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
