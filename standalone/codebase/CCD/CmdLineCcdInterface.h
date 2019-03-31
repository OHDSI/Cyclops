/*
 * CmdLineCcdInterface.h
 *
 *  Created on: April 15, 2014
 *      Author: Marc Suchard
 */

#ifndef CMD_LINE_CCD_INTERFACE_H_
#define CMD_LINE_CCD_INTERFACE_H_

#include "CcdInterface.h"

namespace bsccs {

class CmdLineCcdInterface : public CcdInterface {

public:

    CmdLineCcdInterface(int argc, char* argv[]);
    virtual ~CmdLineCcdInterface();

    void parseCommandLine(std::vector<std::string>& args);

    void parseCommandLine(
            int argc,
            char* argv[],
            CCDArguments &arguments);
            
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

}; // class CmdLineCcdInterface

} // namespace

#endif /* CMD_LINE_CCD_INTERFACE_H_ */
