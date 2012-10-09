/*
* BBROutputWriter.h
*
*  Created on: Aug 27, 2012
*      Author: Sushil Mittal
*/

#ifndef BBROUTPUTWRITER_H_
#define BBROUTPUTWRITER_H_

#include "ModelData.h"
#include "imputation/ImputationPolicy.h"

using namespace std;

class BBROutputWriter{
public:
	BBROutputWriter() {} ;
	virtual ~BBROutputWriter() {}

	void BBROutputWriter::writeFile(const char* fileName, ModelData* modelData) {
		
		int nRows = modelData->getNumberOfRows();
		int nCols = modelData->getNumberOfColumns();

		CompressedDataMatrix* dataTranspose = modelData->transpose();

		ofstream out;
		out.open(fileName,ios::out);

		//map<DrugIdType,int> indexToDrugIdMap = modelData->getDrugNameMap();
		vector<real> y = modelData->getYVectorRef();
		vector<real> z = modelData->getZVectorRef();
		for(int i = 0; i < nRows; i++){

			out << y[i];
			if((int)z.size() > 0)
				out << ":" << z[i];

			int* column;
			real* data;
			int nEntries;
			FormatType formatType = dataTranspose->getFormatType(i);
			switch (formatType)
			{
			case INDICATOR:
//				column = dataTranspose->getCompressedColumnVector(i);
				nEntries = dataTranspose->getNumberOfEntries(i);
				for(int j = 1; j < nEntries; j++){
					out << " " << dataTranspose->getColumn(j).getNumericalLabel() << ":1";
				}
				break;
			case DENSE:
				data = dataTranspose->getDataVector(i);
				for(int j = 1; j < nCols; j++){
					out << " " << modelData->getColumn(j).getNumericalLabel() << ":" << data[j];
				}
				break;
			case SPARSE:
				column = dataTranspose->getCompressedColumnVector(i);
				data = dataTranspose->getDataVector(i);
				nEntries = dataTranspose->getNumberOfEntries(i);
				for(int j = 1; j < nEntries; j++){
					out << " " << modelData->getColumn(column[j]).getNumericalLabel() << ":" << data[j];
				}
				break;
			}
			out << endl;
		}
		if(dataTranspose)
			delete dataTranspose;
	}
};

#endif /* BBROUTPUTWRITER_H_ */
