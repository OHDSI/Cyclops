#ifndef MYHEADER_H_
#define MYHEADER_H_

#ifdef _WIN32
    #include <windows.h>
#endif

#include <string>
#include <sqlext.h>
#pragma comment( lib, "odbc32.lib" )

namespace bsccs 
{

using namespace std;
/////////////////////////

class OutputLocation
{
public:
	OutputLocation()
	{
	}

	virtual void open() = 0;

	virtual void close() = 0;

	virtual void writeLine(string line) = 0;
};

class DataSource 
{
public:
	DataSource() 
	{
	}

	virtual void open(const char* fileName) = 0;

	virtual bool getLine(string& result) = 0;

	virtual char peek() = 0;

	virtual void close() = 0;
};

class FileDataSource : public DataSource 
{
public:
	FileDataSource();
	
	virtual void open(const char* fileName);
	
	virtual bool getLine(string& result);

	virtual char peek();
	
	virtual void close(); 

private:
	ifstream _file;
	const char* _fileName;
};

class FileOutputLocation : public OutputLocation
{
public:
	FileOutputLocation(const char* fileName);
	
	virtual void open();
	
	virtual void close();
	
	virtual void writeLine(string line);
	
private:
	const char* _fileName;
	ofstream _outLog;
};

class OdbcConnection 
{
public:
	OdbcConnection(const char* odbcConnectionName, const char* selectQuery, const char* insertLocation);
	
	bool open();

	void close();

	bool getLine(string &line);

	bool writeLine(string line);
	
private:
	const char* _odbcConectionName;
	const char* _selectQuery;
	const char* _insertLocation;
	bool _isReadingStarted;

	SQLHENV _henv;
	SQLHDBC _hdbc;
	SQLHSTMT _hstmt;
	static const int _bufferSize = 1024;
	vector<unsigned char> _lineData;
	SQLCHAR _buffer[_bufferSize];

	char _insertQuery[1024*100];	
	SQLCHAR  _SqlState[6], _Msg[SQL_MAX_MESSAGE_LENGTH];

	bool isSuccess(SQLRETURN retcode);
	bool startReadingOperation();
};

class OdbcDataSource : public DataSource 
{
public:
	OdbcDataSource(const char* odbcConectionName, const char* selectQuery);
		
	virtual void open(const char* fileName);
	
	virtual void close();
	
	virtual bool getLine(string& result);

	virtual char peek();
		
private:
	bool _returnStoredLine;
	string _storedTextLine;

	OdbcConnection* connection;
};


class OdbcOutputLocation : public OutputLocation
{
public:
	OdbcOutputLocation(const char* connectionName, const char* insertLocation);
	
	virtual void open();
	
	virtual void close();
	
	virtual void writeLine(string line);
	
private:
	OdbcConnection* connection;
};

/////////////////////////

}

#endif /* INPUTREADER_H_ */
