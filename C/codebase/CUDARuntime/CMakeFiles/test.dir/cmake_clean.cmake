FILE(REMOVE_RECURSE
  "./test_generated_Crap.cu.o"
  "test.pdb"
  "test"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/test.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
