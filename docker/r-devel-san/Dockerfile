FROM rocker/r-devel-san

ENV UBSAN_OPTIONS print_stacktrace=1
ENV ASAN_OPTIONS alloc_dealloc_mismatch=0:detect_leaks=0:detect_odr_violation=0:malloc_context_size=10:fast_unwind_on_malloc=false

RUN apt-get -qq update \
  && apt-get -qq dist-upgrade -y \
  && apt-get -qq install git pandoc pandoc-citeproc libssl-dev -y
  
RUN Rscript -e 'install.packages(c("Cyclops","devtools","Matrix"), dependencies = TRUE)'
RUN Rscript -e 'install.packages(c("lattice"), dependencies = TRUE)'

RUN chmod -R a+rw /usr/local/lib/R/site-library

ENV HOME /home/user
RUN useradd --create-home --home-dir $HOME user \
  && chown -R user:user $HOME
WORKDIR $HOME
USER user

RUN mkdir -p ~/.R \
  && echo 'PKG_CXXFLAGS= -fno-omit-frame-pointer -g -Wno-ignored-attributes -Wno-deprecated-declarations -Wno-sign-compare' > ~/.R/Makevars

RUN git clone https://github.com/OHDSI/Cyclops \
  && RD CMD build Cyclops --no-build-vignettes 

# To execute check:
# $ RD CMD check Cyclops*.tar.gz
