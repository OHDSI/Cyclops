/*
 * Timing.h
 *
 *  Created on: Jan 16, 2015
 *      Author: msuchard
 */

#ifndef TIMING_H_
#define TIMING_H_

// Set-up C++11 clock profiling support since Travis-CI does not yet have std::chrono::steady_clock

#include <ctime>
#include <chrono>

namespace bsccs {
    namespace chrono {

//        typedef std::chrono::steady_clock steady_clock;   
        typedef std::chrono::high_resolution_clock steady_clock; // Travis does not support steady_clock
        
//         using TimingUnits = std::chrono::nanoseconds;
        typedef std::chrono::nanoseconds TimingUnits; // Travis-CI does not support using aliases
    	                
        using std::chrono::system_clock; 
        using std::chrono::duration_cast; 
        using std::chrono::duration;
    
    } // namespace chrono
} /* namespace bsccs */

#endif /* TIMING_H_ */
