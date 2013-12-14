#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "InputOutputSystem.h"

namespace bsccs 
{

using namespace std;
/////////////////////////
//FileDataSource

FileDataSource :: FileDataSource() : DataSource() 
{
}
	
void FileDataSource::open(const char* fileName) 
{
	this->_fileName = fileName;
	_file.open(_fileName, std::ifstream::in);
	if (!_file) {
		cout << "Unable to open " << _fileName << endl;
		exit(-1);
	}
}

bool FileDataSource::getLine(string& result)
{
	string line;
		
	bool res = getline(_file, line);
	result = line;
	return res;
}

char FileDataSource::peek()
{
	return _file.peek();
}

void FileDataSource::close() 
{
}

//FileOutputLocation

FileOutputLocation::FileOutputLocation(const char* fileName)
{
	_fileName = fileName;
}

void FileOutputLocation::open()
{
	_outLog.open(_fileName);
	if (!_outLog) {
		cerr << "Unable to open log file: " << _fileName << endl;
		exit(-1);
	}
}

void FileOutputLocation::close()
{
	_outLog.flush();
	_outLog.close();
}

void FileOutputLocation::writeLine(string line)
{
	_outLog << line;
}

//OdbcConnection

OdbcConnection::OdbcConnection(const char* odbcConnectionName,  const char* selectQuery, const char* insertLocation)
{
	_odbcConectionName = odbcConnectionName;
	_selectQuery = selectQuery;
	_insertLocation = insertLocation;
	_isReadingStarted = false;
	_lineData.reserve(_bufferSize);
}

bool OdbcConnection::open()
{
	SQLRETURN retcode;
	SQLCHAR * OutConnStr = (SQLCHAR * )malloc(255);
	SQLSMALLINT * OutConnStrLen = (SQLSMALLINT *)malloc(255);

	retcode = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &_henv);
	if (!isSuccess(retcode))
	{
		cout << "Cannot allocate sql environment handle: " << retcode << endl;
		close();
		return false;
	}

	retcode = SQLSetEnvAttr(_henv, SQL_ATTR_ODBC_VERSION, (void*)SQL_OV_ODBC3, 0); 
	if (!isSuccess(retcode)) 
	{
		cout << "Cannot allocate connection handle: " << retcode << endl;
		close();
		return false;
	}

	retcode = SQLAllocHandle(SQL_HANDLE_DBC, _henv, &_hdbc); 
	if (!isSuccess(retcode)) 
	{
		cout << "Cannot allocate odbc handle: " << retcode << endl;
		close();
		return false;
	}

	SQLSetConnectAttr(_hdbc, SQL_LOGIN_TIMEOUT, (SQLPOINTER)5, 0);
	retcode = SQLConnect(_hdbc, (SQLCHAR*) _odbcConectionName, SQL_NTS, (SQLCHAR*) NULL, 0, NULL, 0);
	if (!isSuccess(retcode)) 
	{
		cout << "Cannot connect to odbc source: " << retcode << endl;
		close();
		return false;
	}
			
	retcode = SQLAllocHandle(SQL_HANDLE_STMT, _hdbc, &_hstmt); 
	if (!isSuccess(retcode)) 
	{
		cout << "Cannot alloc SQL_HANDLE_STMT: " << retcode << endl;
		close();
		return false;
	}

	free(OutConnStr);
	free(OutConnStrLen);
	return true;
}

void OdbcConnection::close()
{
	SQLFreeHandle(SQL_HANDLE_STMT, _hstmt);
	SQLDisconnect(_hdbc);
	SQLFreeHandle(SQL_HANDLE_DBC, _hdbc);
	SQLFreeHandle(SQL_HANDLE_ENV, _henv);
}

bool OdbcConnection::getLine(string &line) 
{
	SQLRETURN    retcode;
	SQLLEN   cbTextLength;

	if (_selectQuery == "") 
	{
		cerr << "Connection doesn't support select opperation";
		return false;
	}

	if (!_isReadingStarted)
	{
		if(!startReadingOperation())
		{
			return false;
		}

		_isReadingStarted = true;
	}

	retcode = SQLFetch(_hstmt);
	if (!isSuccess(retcode)) 
	{
		cout << "Error reading sql data or no more records: " << retcode << endl;
		return false;
	}

	_lineData.clear();
	do 
	{
		retcode = SQLGetData(_hstmt, 1, SQL_C_CHAR, _buffer, _bufferSize, &cbTextLength);
		int size = std::min(cbTextLength, static_cast<SQLLEN>(sizeof(_buffer) / sizeof(SQLCHAR) - 2));
		_lineData.insert( _lineData.end(), _buffer, _buffer + size );
	}
	while(retcode == SQL_SUCCESS_WITH_INFO);

	_lineData.push_back(0);
	line = (const char*)_lineData.data();

	return true;
}

bool OdbcConnection::startReadingOperation()
{
	SQLRETURN    retcode;
	retcode = SQLExecDirect(_hstmt, (SQLCHAR*)_selectQuery, SQL_NTS);
	if (retcode != SQL_SUCCESS) 
	{
		cout << "Error executing sql script: " << retcode << endl;
		close();
		return false;
	}

	return true;
}

bool OdbcConnection::writeLine(string line)
{
	if (_insertLocation == "") 
	{
		cerr << "Connection doesn't support insert opperation";
		return false;
	}

	SQLRETURN retcode, rc2;
	SQLSMALLINT i = 0, MsgLen;
	SQLINTEGER  NativeError;

	sprintf(_insertQuery,"INSERT INTO %s VALUES ('%s')\0", _insertLocation, line.c_str());
	retcode = SQLExecDirect(_hstmt, (SQLCHAR*)_insertQuery, SQL_NTS);
	if (SQL_SUCCESS != retcode)
	{
		rc2 = SQLGetDiagRec(SQL_HANDLE_STMT, _hstmt, i, _SqlState, &NativeError, _Msg, sizeof(_Msg), &MsgLen);
		cout << endl << "Error writing data in table " << endl << "SqlState=" << _SqlState << " Native Error=" << NativeError << endl;
		cout << _Msg << endl;
		return false;
    }

	return true;
}

bool OdbcConnection::isSuccess(SQLRETURN retcode)
{
	return retcode == SQL_SUCCESS || retcode == SQL_SUCCESS_WITH_INFO;
}


//OdbcDataSource


OdbcDataSource::OdbcDataSource(const char* odbcConectionName, const char* selectQuery) : DataSource() 
{
	_returnStoredLine = false;
	//connection = new OdbcConnection("TestSCCS", "SELECT SampleInput FROM TestInput", "");
	connection = new OdbcConnection(odbcConectionName, selectQuery, "");
}
	
void OdbcDataSource::open(const char* fileName) 
{
	if(!connection->open())
	{
		exit(-1);
	}
}

void OdbcDataSource::close() 
{
	connection->close();
	if (connection) 
	{
		delete connection;
	}
}

bool OdbcDataSource::getLine(string& result)
{
	if (_returnStoredLine) 
	{
		result = _storedTextLine;
		_returnStoredLine = false;
		return true;
	}

	string line;
	if(!connection->getLine(line))
	{
		return false;
	}

	result = line;
	return true;
}

char OdbcDataSource::peek()
{
	if (!getLine(_storedTextLine))
	{
		exit(-1);
	}
		
	_returnStoredLine = true;
	char result = _storedTextLine[0];
	return result;
}

//OdbcOutputLocation

OdbcOutputLocation::OdbcOutputLocation(const char* connectionName, const char* insertQuery)
{
	connection = new OdbcConnection(connectionName, "", insertQuery);
}
	
void OdbcOutputLocation::open()
{
	if(!connection->open())
	{
		exit(-1);
	}
}
	
void OdbcOutputLocation::close()
{
	connection->close();
	if (connection) 
	{
		delete connection;
	}
}
	
void OdbcOutputLocation::writeLine(string line)
{
	connection->writeLine(line);
}

/////////////////////////

}