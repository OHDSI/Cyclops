/*
 * RcppCcdInterface.h
 *
 * @author Marc Suchard
 */

#ifndef RCPP_CCD_INTERFACE_H_
#define RCPP_CCD_INTERFACE_H_

#include "CcdInterface.h"

namespace bsccs {

class RcppCcdInterface : public CcdInterface {

public:

    RcppCcdInterface();
    virtual ~RcppCcdInterface();
            
protected:            
            
    void initializeModelImpl(
            ModelData** modelData,
            CyclicCoordinateDescent** ccd,
            AbstractModelSpecifics** model);
            
    void predictModelImpl(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData); 
            
    void logModelImpl(
            CyclicCoordinateDescent *ccd,
            ModelData *modelData,
            ProfileInformationMap& profileMap,
            bool withASE); 
            
     void diagnoseModelImpl(
            CyclicCoordinateDescent *ccd, 
            ModelData *modelData,	
    		double loadTime,
    		double updateTime);  
}; // class RcppCcdInterface

} // namespace

#endif /* CCD_H_ */
