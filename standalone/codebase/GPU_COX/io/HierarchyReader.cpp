	/*
 * HierarchyReader.cpp
 *
 *  Created on: Jun 7, 2011
 *      Author: tshaddox
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "HierarchyReader.h"


#define MAX_ENTRIES		1000000000

#define FORMAT_MATCH_1	"\"ingredientIds\""
#define MATCH_LENGTH_1	15


#define MISSING_STRING	"NA"

#ifdef USE_DRUG_STRING
	#define NO_DRUG			"0"
#else
	#define	NO_DRUG			0
#endif

using namespace std;

namespace bsccs {

HierarchyReader::HierarchyReader(const char* fileName, AbstractModelData* modelData) {
	//cout << "fileName = " << fileName << endl;
	ifstream in(fileName);
	if (!in) {
		cerr << "Unable to open " << fileName << endl;
		exit(-1);
	}

	string line;
	getline(in, line); // Read header

	if (line.compare(0, MATCH_LENGTH_1, FORMAT_MATCH_1) != 0) {
		cerr << "Unrecognized file type" << endl;
	//	exit(-1);
	}

	int currentEntry = 0;

	while (getline(in, line) && (currentEntry < MAX_ENTRIES)) {
		int parent;
		int child;

		string meaningless;
		if (!line.empty()) {
			stringstream ss(line.c_str());
			ss >> child; //subnode in the hierarchy

			if (modelData->getColumnIndexByName(child) > -1){
				child = modelData->getColumnIndexByName(child);
				ss >> parent; //supernode in the hierarchy

				if (parent == 0) { //if not associated with a group, the sub- and supernodes have same id num
					getChildMap[child].push_back(child);
					getParentMap[child] = child;
				} else {
					getChildMap[parent].push_back(child);
					getParentMap[child] = parent;
				}
			}
		}
	}
}

void HierarchyReader::printChildren(int parent){
	vector<int> children = getChildMap[parent];
	cout << "Children of " << parent << " are <";
	for(int i = 0; i < children.size(); i++){
		cout << children[i] << ", ";
	}
	cout << "> " << endl;
}


HierarchicalChildMap HierarchyReader::returnGetChildMap(){
	return(getChildMap);
}

HierarchicalParentMap HierarchyReader::returnGetParentMap(){
	return(getParentMap);
}


vector<int> HierarchyReader::getChildren(int parent){
	return(getChildMap[parent]);
}


int HierarchyReader::getParent(int child){
	return(getParentMap[child]);
}


}

