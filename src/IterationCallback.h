#pragma once

///////////////////////////////////////////////////////////////////////////////

class IIterationCallback {
public:
	virtual ~IIterationCallback() {}

	// Нужно звать до начала итерации.
	// Возвращает true, если нужно продолжать выполнение итерации.
	virtual bool BeginIteration() = 0;

	// Нужно звать после выполнения итерации.
	virtual void EndIteration( const NumericType difference ) = 0;
};

///////////////////////////////////////////////////////////////////////////////

class CSimpleIterationCallback : public IIterationCallback {
public:
	explicit CSimpleIterationCallback( const NumericType eps = DefaultEps ) :
		eps( eps ),
		difference( numeric_limits<NumericType>::max() )
	{
	}

	virtual bool BeginIteration()
	{
		return ( !( difference < eps ) );
	}
	virtual void EndIteration( const NumericType _difference )
	{
		difference = _difference;
	}

private:
	const NumericType eps;
	NumericType difference;
};

///////////////////////////////////////////////////////////////////////////////

class CIterationCallback : public CSimpleIterationCallback {
public:
	CIterationCallback( ostream& outputStream, const size_t id,
		const NumericType eps = DefaultEps,
		const size_t iterationsLimit = numeric_limits<size_t>::max() ) :
		CSimpleIterationCallback( eps ),
		out( outputStream ),
		id( id ),
		iterationsLimit( iterationsLimit ),
		iteration( 0 )
	{
	}

	virtual bool BeginIteration()
	{
		if( !CSimpleIterationCallback::BeginIteration()
			|| !( iteration < iterationsLimit ) ) {
			return false;
		}

		out << "(" << id << ") Iteratition #" << iteration << " started." << endl;
		return true;
	}

	virtual void EndIteration( const NumericType diff )
	{
		CSimpleIterationCallback::EndIteration( diff );

		cout << "(" << id << ") Iteratition #" << iteration << " finished "
			<< "with difference `" << diff << "`." << endl;
		iteration++;
	}

private:
	ostream& out;
	const size_t id;
	const size_t iterationsLimit;
	size_t iteration;
};

///////////////////////////////////////////////////////////////////////////////
