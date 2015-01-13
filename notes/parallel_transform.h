namespace Rcpp{
    namespace parallel{
    
        #if defined(RCPP11_EXPERIMENTAL_PARALLEL)
        template <typename InputIterator, typename OutputIterator, typename Function>
        void transform( int nthreads, InputIterator begin, InputIterator end, OutputIterator target, Function fun ){ 
            std::vector<std::thread> workers(nthreads-1) ;
            R_xlen_t chunk_size = std::distance(begin, end) / nthreads ;
            R_xlen_t start=0; 
            for( int i=0; i<nthreads-1; i++, start+=chunk_size){
                workers[i] = std::thread( std::transform<InputIterator, OutputIterator, Function>, 
                    begin + start, begin + start + chunk_size, 
                    target + start, 
                    fun) ;   
            }
            std::transform( begin + start, end, target + start, fun ) ;
            for( int i=0; i<nthreads-1; i++) workers[i].join() ;
        }
        #else
        template <typename InputIterator, typename OutputIterator, typename Function>
        inline void transform( int, InputIterator begin, InputIterator end, OutputIterator target, Function fun ){ 
            std::transform( begin, end, target, fun ) ;
        }
        #endif
    }   
}


template <class T, 
          class InputIterator, 
          class MapFunction, 
          class ReductionFunction>
T MapReduce_n(InputIterator in, 
              unsigned int size, 
              T baseval, 
              MapFunction mapper, 
              ReductionFunction reducer)
{
    T val = baseval;

    #pragma omp parallel
    {
        T map_val = baseval;

        #pragma omp for nowait
        for (auto i = 0U; i < size; ++i)
        {
            map_val = reducer(map_val, mapper(*(in + i)));
        }

        #pragma omp critical
        val = reducer(val, map_val);
    }

    return val;
}