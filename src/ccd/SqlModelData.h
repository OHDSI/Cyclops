/*
 * SqlModelData.h
 *
 *  Created on: Jun 12, 2014
 *      Author: msuchard
 */

#ifndef SQLMODELDATA_H_
#define SQLMODELDATA_H_

#include "ModelData.h"

namespace bsccs {


class SqlModelData : public ModelData {

public:
	SqlModelData();
	
	virtual ~SqlModelData();

private:
	// Disable copy-constructors and copy-assignment
	SqlModelData(const ModelData&);
	SqlModelData& operator = (const SqlModelData&);
};

} /* namespace bsccs */
#endif /* SQLMODELDATA_H_ */
