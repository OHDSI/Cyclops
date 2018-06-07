FROM rocker/r-devel-ubsan-clang

ENV UBSAN_OPTIONS print_stacktrace=1
ENV ASAN_OPTIONS alloc_dealloc_mismatch=0:detect_leaks=0:detect_odr_violation=0:malloc_context_size=10:fast_unwind_on_malloc=false

RUN apt-get -qq update \
#  && apt-get -qq dist-upgrade -y \
  && apt-get -qq install git pandoc pandoc-citeproc libssl-dev  -y

RUN apt-get -qq install libgmp-dev libmpfr-dev libxml2-dev -y

## Set default CRAN repo
RUN echo 'options(download.file.method="wget")' >> /usr/local/lib/R/etc/Rprofile.site

RUN chmod -R a+rw /usr/local/lib/R/site-library

ENV HOME /home/user
RUN useradd --create-home --home-dir $HOME user \
  && chown -R user:user $HOME
WORKDIR $HOME
USER user

# RUN mkdir -p ~/.R \
#   && echo 'PKG_CXXFLAGS= -I../inst/include -stdlib=libc++ -fno-omit-frame-pointer -g -Wno-ignored-attributes -Wno-deprecated-declarations -Wno-sign-compare' > ~/.R/Makevars \
#   && echo 'MAIN_LD="clang++-4.0 -static-asan -fsanitize=undefined,address"' >> ~/.R/Makevars
#
# RUN RDscript -e 'install.packages("Rcpp")'

RUN mkdir -p ~/.R \
  && echo 'PKG_CXXFLAGS= -I../inst/include -stdlib=libc++ -fno-omit-frame-pointer -g -Wno-ignored-attributes -Wno-deprecated-declarations -Wno-sign-compare' > ~/.R/Makevars \
  && echo 'CXX=clang++-4.0 -stdlib=libc++' >> ~/.R/Makevars \
  && echo 'CXX11=clang++-4.0 -stdlib=libc++' >> ~/.R/Makevars \
  && echo 'MAIN_LD="clang++-4.0 -static-asan -fsanitize=undefined,address"' >> ~/.R/Makevars

RUN RDscript -e 'install.packages("Rcpp")'

RUN RDscript -e 'install.packages("Matrix")'
RUN RDscript -e 'install.packages("igraph")'
RUN RDscript -e 'install.packages("expm")'
RUN RDscript -e 'install.packages("lattice")'

RUN RDscript -e 'install.packages("devtools")'
RUN RDscript -e 'install.packages("RcppParallel")'
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

RUN mkdir -p ~/.R \
  && echo 'PKG_CXXFLAGS= -I../inst/include -stdlib=libc++ -fno-omit-frame-pointer -g -Wno-ignored-attributes -Wno-deprecated-declarations -Wno-sign-compare' > ~/.R/Makevars \
  && echo 'CXX11=clang++-4.0 -stdlib=libc++ -fsanitize=address,undefined -fno-sanitize=float-divide-by-zero -fno-omit-frame-pointer -fsanitize-address-use-after-scope' >> ~/.R/Makevars \
  && echo 'MAIN_LD="clang++-4.0 -static-asan -fsanitize=undefined,address"' >> ~/.R/Makevars

RUN RDscript -e 'install.packages("Cyclops", dependencies = TRUE)'

#RUN git clone https://github.com/OHDSI/Cyclops \
#  && RD CMD build Cyclops --no-build-vignettes

ENV UBSAN_OPTIONS print_stacktrace=1
ENV ASAN_OPTIONS alloc_dealloc_mismatch=0:detect_leaks=0:detect_odr_violation=0:malloc_context_size=10:fast_unwind_on_malloc=false

# To execute check: docker run --cap-add SYS_PTRACE -v /Users/msuchard/Dropbox/Projects/Cyclops:/home/user/Cyclops --rm -it cyclops-san-clang /usr/bin/bash
# $ RD CMD check Cyclops*.tar.gz

