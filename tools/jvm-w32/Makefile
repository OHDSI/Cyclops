# Adapted from: https://github.com/s-u/rJava.git

# helper tools and libs for building and running rJava for Windows
# Author: Simon Urbanek
#         with contributions from Brian Ripley

include $(R_HOME)/etc$(R_ARCH)/Makeconf

TARGETS=findjava.exe

# findjava.exe    - helper tool to find the current JDK from the registry

all: $(TARGETS)

# detect 64-bit Windows
ifeq ($(strip $(shell $(R_HOME)/bin/R --slave -e 'cat(.Machine$$sizeof.pointer)')),8)
JVM64DEF=64
endif

# compile findjava.exe from source - no magic here, no special libs necessary
findjava.o: findjava.c
	$(CC) -O2 -c -o $@ $^

findjava.exe: findjava.o
	$(CC) -s -o $@ $^

# just cleanup everything
clean:
	rm -f *.o *~ $(TARGETS)

.PHONY: all clean
