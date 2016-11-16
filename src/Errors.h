#pragma once

///////////////////////////////////////////////////////////////////////////////

class CException : public exception {
public:
	explicit CException( const string& errorText = "" ) :
		text( errorText )
	{
	}
	virtual ~CException() throw()
	{
	}

	virtual const char* what() const throw()
	{
		return text.c_str();
	}

protected:
	void SetErrorText( const string& errorText )
	{
		text = errorText;
	}

private:
	string text;
};

///////////////////////////////////////////////////////////////////////////////
