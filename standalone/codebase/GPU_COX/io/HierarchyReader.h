/*
 * HierarchyReader.h
 *
 *  Created on: Jun 7, 2011
 *      Author: tshaddox
 */

#ifndef HIERARCHYREADER_H_
#define HIERARCHYREADER_H_

#include <iostream>
#include <fstream>

#include <vector>
#include <map>

using namespace std;

#include "Types.h"
#include "io/InputReader.h"

//#define USE_DRUG_STRING

namespace bsccs {

#ifdef USE_DRUG_STRING
	typedef string DrugIdType; // TODO String do not get sorted in numerical order
#else
	typedef int DrugIdType;
#endif

class HierarchyReader {
public:
	HierarchyReader();

	HierarchyReader(const char* fileName, AbstractModelData* modelData);

	HierarchicalChildMap returnGetChildMap();

	HierarchicalParentMap returnGetParentMap();

	void printChildren(int parent);

	vector<int> getChildren(int parent);

	int getParent(int child);

	std::vector<vector<int> > getChildMap;
	std::vector<int> getParentMap;
	std::map<int, int> drugIdToIndex;

private:


};

}

#endif /* HIERARCHYREADER_H_ */
