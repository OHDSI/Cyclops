FROM rocker/r-devel-ubsan-clang

ENV UBSAN_OPTIONS print_stacktrace=1
ENV ASAN_OPTIONS alloc_dealloc_mismatch=0:detect_leaks=0:detect_odr_violation=0:malloc_context_size=10:fast_unwind_on_malloc=false

RUN apt-get -qq update \
  && apt-get -qq remove r-base-core r-cran-* -y \
  && apt-get -qq dist-upgrade -y \
  && apt-get -qq install git pandoc pandoc-citeproc libssl-dev  -y

RUN apt-get -qq install libgmp-dev libmpfr-dev libxml2-dev -y

RUN apt-get -qq install clang-6.0 -y

## Build Rdevel with clang-6.0

## Add symlink and check out R-devel
RUN rm /usr/local/bin/llvm-symbolizer \
    && ln -s $(which llvm-symbolizer-6.0) /usr/local/bin/llvm-symbolizer \
	&& cd /tmp \
	&& svn co https://svn.r-project.org/R/trunk R-devel

## Build and install according extending the standard 'recipe' I emailed/posted years ago
## Leak detection does not work at build time, see https://github.com/google/sanitizers/issues/764 and the fact that we cannot add privileges during build (e.g. https://unix.stackexchange.com/q/329816/19205)
RUN cd /tmp/R-devel \
	&& R_PAPERSIZE=letter \
	   R_BATCHSAVE="--no-save --no-restore" \
	   R_BROWSER=xdg-open \
	   PAGER=/usr/bin/pager \
	   PERL=/usr/bin/perl \
	   R_UNZIPCMD=/usr/bin/unzip \
	   R_ZIPCMD=/usr/bin/zip \
	   R_PRINTCMD=/usr/bin/lpr \
	   LIBnn=lib \
	   AWK=/usr/bin/awk \
	   CC="clang-6.0 -fsanitize=address,undefined -fno-sanitize=float-divide-by-zero -fno-omit-frame-pointer -fsanitize-address-use-after-scope" \
	   CXX="clang++-6.0 -stdlib=libc++ -fsanitize=address,undefined -fno-sanitize=float-divide-by-zero -fno-omit-frame-pointer -fsanitize-address-use-after-scope" \
	   CFLAGS="-g -O3 -Wall -pedantic -mtune=native" \
	   FFLAGS="-g -O2 -mtune=native" \
	   FCFLAGS="-g -O2 -mtune=native" \
	   CXXFLAGS="-g -O3 -Wall -pedantic -mtune=native" \
	   MAIN_LD="clang++-6.0 -stdlib=libc++ -fsanitize=undefined,address" \
	   FC="gfortran" \
	   F77="gfortran" \
	   ASAN_OPTIONS=detect_leaks=0 \
	   ./configure --enable-R-shlib \
	       --without-blas \
	       --without-lapack \
	       --with-readline \
	       --without-recommended-packages \
	       --program-suffix=dev \
	       --disable-openmp \
	&& ASAN_OPTIONS=detect_leaks=0 make \
	&& ASAN_OPTIONS=detect_leaks=0 make install \
	&& ASAN_OPTIONS=detect_leaks=0 make clean

## Set default CRAN repo
RUN echo 'options(download.file.method="wget")' >> /usr/local/lib/R/etc/Rprofile.site

RUN chmod -R a+rw /usr/local/lib/R/site-library

ENV HOME /home/user
RUN useradd --create-home --home-dir $HOME user \
  && chown -R user:user $HOME
WORKDIR $HOME
USER user

RUN mkdir -p ~/.R \
  && echo 'PKG_CXXFLAGS= -I../inst/include -stdlib=libc++ -fno-omit-frame-pointer -g -Wno-ignored-attributes -Wno-deprecated-declarations -Wno-sign-compare' > ~/.R/Makevars \
  && echo 'CXX=clang++-6.0 -stdlib=libc++' >> ~/.R/Makevars \
  && echo 'CXX11=clang++-6.0 -stdlib=libc++' >> ~/.R/Makevars

RUN RDscript -e 'install.packages("Rcpp")'

RUN RDscript -e 'install.packages("Matrix")'
RUN RDscript -e 'install.packages("igraph")'
RUN RDscript -e 'install.packages("expm")'
RUN RDscript -e 'install.packages("lattice")'

RUN RDscript -e 'install.packages("devtools")'
#RUN RDscript -e 'install.packages("RcppParallel")'
RUN RDscript -e 'install.packages("RcppEigen")'
RUN RDscript -e 'install.packages("microbenchmark")'
RUN RDscript -e 'install.packages("stringi")'
RUN RDscript -e 'install.packages("MASS")'
RUN RDscript -e 'install.packages("survival")'
RUN RDscript -e 'install.packages("nnet")'
RUN RDscript -e 'install.packages("gnm")'
RUN RDscript -e 'install.packages("testthat")'
RUN RDscript -e 'install.packages("ggplot2")'
RUN RDscript -e 'install.packages("ff")'
RUN RDscript -e 'install.packages("ffbase")'
RUN RDscript -e 'install.packages("roxygen2")'
RUN RDscript -e 'install.packages("Cyclops", dependencies = TRUE)'

RUN mkdir -p ~/.R \
  && echo 'PKG_CXXFLAGS= -I../inst/include -stdlib=libc++ -fno-omit-frame-pointer -g' > ~/.R/Makevars \
  && echo 'CXX11=clang++-6.0 -stdlib=libc++ -fsanitize=address,undefined -fno-sanitize=float-divide-by-zero -fno-omit-frame-pointer -fsanitize-address-use-after-scope' >> ~/.R/Makevars

#RUN git clone https://github.com/OHDSI/Cyclops \
#  && RD CMD build Cyclops --no-build-vignettes

ENV UBSAN_OPTIONS print_stacktrace=1
ENV ASAN_OPTIONS alloc_dealloc_mismatch=0:detect_leaks=0:detect_odr_violation=0:malloc_context_size=10:fast_unwind_on_malloc=false

# To build container: docker build -t cyclops-san-clang .
# To execute check: docker run --cap-add SYS_PTRACE -v /Users/msuchard/Dropbox/Projects/Cyclops:/home/user/Cyclops --rm -it cyclops-san-clang /usr/bin/bash
# $ RD CMD check Cyclops*.tar.gz

