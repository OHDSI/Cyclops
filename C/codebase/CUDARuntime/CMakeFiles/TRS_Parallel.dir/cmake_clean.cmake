FILE(REMOVE_RECURSE
  "./TRS_Parallel_generated_CUDASparseRowVector.cu.o"
  "libTRS_Parallel.pdb"
  "libTRS_Parallel.a"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/TRS_Parallel.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
