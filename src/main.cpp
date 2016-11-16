#include <cmath>
#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

#include <MpiSupport.h>

///////////////////////////////////////////////////////////////////////////////

typedef double NumericType;
const MPI_Datatype MpiNumericType = MPI_DOUBLE;
const NumericType DefaultEps = static_cast<NumericType>( 0.0001 );

///////////////////////////////////////////////////////////////////////////////

inline NumericType F( NumericType x, NumericType y )
{
	const NumericType xy2 = ( x + y ) * ( x + y );
	const NumericType f = 4 * ( 1 - 2 * xy2 ) * exp( 1 - xy2 );
	return f;
}

inline NumericType Phi( NumericType x, NumericType y )
{
	const NumericType xy2 = ( x + y ) * ( x + y );
	const NumericType phi = exp( 1 - xy2 );
	return phi;
}

///////////////////////////////////////////////////////////////////////////////

class IIterationCallback {
public:
	virtual ~IIterationCallback() = 0 {}

	// Нужно звать до начала итерации.
	// Возвращает true, если нужно продолжать выполнение итерации.
	virtual bool BeginIteration() = 0;

	// Нужно звать после выполнения итерации.
	virtual void EndIteration( const NumericType difference ) = 0;
};

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
			|| !( iteration < iterationsLimit ) )
		{
			return false;
		}

		out << "(" << id << ") Iteratition #" << iteration << " started." << endl;
		return true;
	}

	virtual void EndIteration( const NumericType difference )
	{
		CSimpleIterationCallback::EndIteration( difference );

		cout << "(" << id << ") Iteratition #" << iteration << " finished "
			<< "with difference `" << difference << "`." << endl;
		iteration++;
	}

private:
	ostream& out;
	const size_t id;
	const size_t iterationsLimit;
	size_t iteration;
};

///////////////////////////////////////////////////////////////////////////////

struct CFraction {
	NumericType Numerator; // числитель
	NumericType Denominator; // знаменатель

	CFraction() :
		Numerator( 0 ),
		Denominator( 0 )
	{
	}

	NumericType Value() const
	{
		return ( Numerator / Denominator );
	}
};

///////////////////////////////////////////////////////////////////////////////

class CMatrix {
public:
	CMatrix() :
		sizeX( 0 ),
		sizeY( 0 )
	{
	}

	CMatrix( size_t sizeX, size_t sizeY )
	{
		Init( sizeX, sizeY );
	}

	CMatrix( const CMatrix& other ) { *this = other; };
	CMatrix& operator=( const CMatrix& other )
	{
		sizeX = other.sizeX;
		sizeY = other.sizeY;
		values = other.values;
		return *this;
	}
	

	void Init( const size_t sizeX, const size_t sizeY )
	{
		this->sizeX = sizeX;
		this->sizeY = sizeY;
		values.resize( sizeX * sizeY );
		fill( values.begin(), values.end(), static_cast<NumericType>( 0 ) );
	}

	NumericType& operator()( size_t x, size_t y )
	{
		return values[y * sizeX + x];
	}
	NumericType operator()( size_t x, size_t y ) const
	{
		return values[y * sizeX + x];
	}

	size_t SizeX() const { return sizeX; }
	size_t SizeY() const { return sizeY; }

private:
	size_t sizeX;
	size_t sizeY;
	vector<NumericType> values;
};

///////////////////////////////////////////////////////////////////////////////

struct CMatrixPart {
	size_t BeginX;
	size_t EndX;
	size_t BeginY;
	size_t EndY;

	CMatrixPart() :
		BeginX( 0 ), EndX( 0 ),
		BeginY( 0 ), EndY( 0 )
	{
	}

	CMatrixPart( size_t beginX, size_t endX, size_t beginY, size_t endY ) :
		BeginX( beginX ), EndX( endX ),
		BeginY( beginY ), EndY( endY )
	{
		assert( BeginX < EndX );
		assert( BeginY < EndY );
	}

	void SetRow( size_t beginX, size_t endX, size_t y )
	{
		assert( beginX < endX );
		BeginX = beginX;
		EndX = endX;
		BeginY = y;
		EndY = y + 1;
	}

	void SetColumn( size_t x, size_t beginY, size_t endY )
	{
		assert( beginY < endY );
		BeginX = x;
		EndX = x + 1;
		BeginY = beginY;
		EndY = endY;
	}

	size_t SizeX() const
	{
		return ( EndX - BeginX );
	}

	size_t SizeY() const
	{
		return ( EndY - BeginY );
	}

	size_t Size() const
	{
		return SizeX() * SizeY();
	}
};

ostream& operator<<( ostream& out, const CMatrixPart& matrixPart )
{
	out << "[" << matrixPart.BeginX << ", " << matrixPart.EndX << ") x "
		<< "[" << matrixPart.BeginY << ", " << matrixPart.EndY << ")";
	return out;
}

///////////////////////////////////////////////////////////////////////////////

class CUniformPartition {
private:
	CUniformPartition( const CUniformPartition& );
	CUniformPartition& operator=( const CUniformPartition& );

public:
	CUniformPartition() {}

	void PartInit( NumericType p0, NumericType pN, size_t size, size_t begin, size_t end )
	{
		if( !( p0 < pN ) ) {
			throw CException( "CUniformPartition: bad interval" );
		}
		if( !( size > 1 ) ) {
			throw CException( "CUniformPartition: inavalid size" );
		}
		if( !( begin < end && end <= size ) ) {
			throw CException( "CUniformPartition: invalid [begin, end)" );
		}

		ps.clear();
		ps.reserve( end - begin );

		for( size_t i = begin; i < end; i++ ) {
			const NumericType part = static_cast<NumericType>( i ) / ( size - 1 );
			const NumericType p = part * pN + ( 1 - part ) * p0;
			ps.push_back( p );
		}
	}

	void Init( NumericType p0, NumericType pN, size_t N )
	{
		PartInit( p0, pN, N, 0, N );
	}

	size_t Size() const
	{
		return ps.size();
	}
	NumericType operator[]( size_t i ) const
	{
		return Point( i );
	}
	NumericType Point( size_t i ) const
	{
		return ps[i];
	}
	NumericType Step( size_t i ) const
	{
		return ( Point( i + 1 ) - Point( i ) );
	}
	NumericType AverageStep( size_t i ) const
	{
		//return ( Step( i ) + Step( i - 1 ) ) / static_cast<NumericType>( 2 );
		return ( Point( i + 1 ) - Point( i - 1 ) ) / static_cast<NumericType>( 2 );
	}

private:
	vector<NumericType> ps;
};

struct CUniformGrid {
	CUniformPartition X;
	CUniformPartition Y;

	CMatrixPart Column( size_t x, size_t decreaseTop = 0, size_t decreaseBottom = 0 ) const
	{
		CMatrixPart part;
		part.SetColumn( x, decreaseTop, Y.Size() - decreaseBottom );
		return part;
	}
	CMatrixPart Row( size_t y, size_t decreaseLeft = 0, size_t decreaseRight = 0 ) const
	{
		CMatrixPart part;
		part.SetRow( decreaseLeft, X.Size() - decreaseRight, y );
		return part;
	}
};

///////////////////////////////////////////////////////////////////////////////

NumericType LaplasOperator( const CMatrix& matrix, const CUniformGrid& grid, size_t x, size_t y )
{
	const NumericType ldx = ( matrix( x, y ) - matrix( x - 1, y ) ) / grid.X.Step( x - 1 );
	const NumericType rdx = ( matrix( x + 1, y ) - matrix( x, y ) ) / grid.X.Step( x );
	const NumericType tdy = ( matrix( x, y ) - matrix( x, y - 1 ) ) / grid.Y.Step( y - 1 );
	const NumericType bdy = ( matrix( x, y + 1 ) - matrix( x, y ) ) / grid.Y.Step( y );
	const NumericType dx = ( ldx - rdx ) / grid.X.AverageStep( x );
	const NumericType dy = ( tdy - bdy ) / grid.Y.AverageStep( y );
	return ( dx + dy );
}

// Вычисление невязки rij во внутренних точках.
void CalcR( const CMatrix&p, const CUniformGrid& grid, CMatrix& r )
{
	for( size_t x = 1; x < r.SizeX() - 1; x++ ) {
		for( size_t y = 1; y < r.SizeY() - 1; y++ ) {
			r( x, y ) = LaplasOperator( p, grid, x, y ) - F( grid.X[x], grid.Y[y] );
		}
	}
}

// Вычисление значений gij во внутренних точках.
void CalcG( const CMatrix&r, const NumericType alpha, CMatrix& g )
{
	for( size_t x = 1; x < g.SizeX() - 1; x++ ) {
		for( size_t y = 1; y < g.SizeY() - 1; y++ ) {
			g( x, y ) = r( x, y ) - alpha * g( x, y );
		}
	}
}

// Вычисление значений pij во внутренних точках, возвращается максимум норма.
NumericType CalcP( const CMatrix&g, const NumericType tau, CMatrix& p )
{
	NumericType difference = 0;
	for( size_t x = 1; x < p.SizeX() - 1; x++ ) {
		for( size_t y = 1; y < g.SizeY() - 1; y++ ) {
			const NumericType newValue = p( x, y ) - tau * g( x, y );
			difference = max( difference, abs( newValue - p( x, y ) ) );
			p( x, y ) = newValue;
		}
	}
	return difference;
}

// Вычисление alpha.
CFraction CalcAlpha( const CMatrix&r, const CMatrix&g, const CUniformGrid& grid )
{
	CFraction alpha;
	for( size_t x = 1; x < r.SizeX() - 1; x++ ) {
		for( size_t y = 1; y < r.SizeY() - 1; y++ ) {
			const NumericType common = g( x, y ) * grid.X.AverageStep( x ) * grid.Y.AverageStep( y );
			alpha.Numerator += LaplasOperator( r, grid, x, y ) * common;
			alpha.Denominator += LaplasOperator( g, grid, x, y ) * common;
		}
	}
	return alpha;
}

// Вычисление tau.
CFraction CalcTau( const CMatrix&r, const CMatrix&g, const CUniformGrid& grid )
{
	CFraction tau;
	for( size_t x = 1; x < r.SizeX() - 1; x++ ) {
		for( size_t y = 1; y < r.SizeY() - 1; y++ ) {
			const NumericType common = g( x, y ) * grid.X.AverageStep( x ) * grid.Y.AverageStep( y );
			tau.Numerator += r( x, y ) * common;
			tau.Denominator += LaplasOperator( g, grid, x, y ) * common;
		}
	}
	return tau;
}

///////////////////////////////////////////////////////////////////////////////

class CExchangeDefinition {
public:
	CExchangeDefinition( size_t rank,
			const CMatrixPart& sendPart,
			const CMatrixPart& recvPart ) :
		rank( rank ),
		sendPart( sendPart ),
		recvPart( recvPart )
	{
		sendBuffer.reserve( sendPart.Size() );
		recvBuffer.reserve( recvPart.Size() );
	}

	const CMatrixPart& SendPart() const { return sendPart; }
	const CMatrixPart& RecvPart() const { return recvPart; }

	void DoExchange( CMatrix& matrix );
	void Wait( CMatrix& matrix );

private:
	size_t rank;

	CMatrixPart sendPart;
	MPI_Request sendRequest;
	vector<NumericType> sendBuffer;

	CMatrixPart recvPart;
	MPI_Request recvRequest;
	vector<NumericType> recvBuffer;
};

///////////////////////////////////////////////////////////////////////////////

void CExchangeDefinition::DoExchange( CMatrix& matrix )
{
	sendBuffer.clear();
	for( size_t x = sendPart.BeginX; x < sendPart.EndX; x++ ) {
		for( size_t y = sendPart.BeginY; y < sendPart.EndY; y++ ) {
			sendBuffer.push_back( matrix( x, y ) );
		}
	}

	MpiCheck( MPI_Isend( sendBuffer.data(), sendBuffer.size(),
		MpiNumericType, rank, 0, MPI_COMM_WORLD, &sendRequest ), "MPI_Isend" );

	recvBuffer.resize( recvPart.Size() );
	MpiCheck( MPI_Irecv( recvBuffer.data(), recvBuffer.size(),
		MpiNumericType, rank, 0, MPI_COMM_WORLD, &recvRequest ), "MPI_Irecv" );
}

void CExchangeDefinition::Wait( CMatrix& matrix )
{
	MpiCheck( MPI_Wait( &recvRequest, MPI_STATUS_IGNORE ), "MPI_Wait" );

	vector<NumericType>::const_iterator value = recvBuffer.begin();
	for( size_t x = recvPart.BeginX; x < recvPart.EndX; x++ ) {
		for( size_t y = recvPart.BeginY; y < recvPart.EndY; y++ ) {
			matrix( x, y ) = *value;
			++value;
		}
	}

	MpiCheck( MPI_Wait( &sendRequest, MPI_STATUS_IGNORE ), "MPI_Wait" );
}

///////////////////////////////////////////////////////////////////////////////

class CExchangeDefinitions : public vector<CExchangeDefinition> {
public:
	CExchangeDefinitions() {}

	void Exchange( CMatrix& matrix )
	{
		for( vector<CExchangeDefinition>::iterator i = begin(); i != end(); ++i ) {
			i->DoExchange( matrix );
		}
		// Сначала начинаем все асинхронные операции, затем ждём.
		for( vector<CExchangeDefinition>::iterator i = begin(); i != end(); ++i ) {
			i->Wait( matrix );
		}
	}
};

///////////////////////////////////////////////////////////////////////////////

void GetBeginEndPoints( const size_t numberOfPoints, const size_t numberOfBlocks,
	const size_t blockIndex, size_t& beginPoint, size_t& endPoint )
{
	const size_t objectsPerProcess = numberOfPoints / numberOfBlocks;
	const size_t additionalPoints = numberOfPoints % numberOfBlocks;
	beginPoint = objectsPerProcess * blockIndex + min( blockIndex, additionalPoints );
	endPoint = beginPoint + objectsPerProcess;
	if( blockIndex < additionalPoints ) {
		endPoint++;
	}
}

///////////////////////////////////////////////////////////////////////////////

class CProgram {
public:
	static void Run( size_t pointsX, size_t pointsY, IIterationCallback& callback );

private:
	const size_t numberOfProcesses;
	const size_t rank;
	const size_t pointsX;
	const size_t pointsY;
	size_t processesX;
	size_t processesY;
	size_t rankX;
	size_t rankY;
	size_t beginX;
	size_t endX;
	size_t beginY;
	size_t endY;
	CExchangeDefinitions exchangeDefinitions;
	CUniformGrid grid;
	CMatrix p;
	CMatrix r;
	CMatrix g;
	NumericType difference;

	CProgram( size_t pointsX, size_t pointsY );

	bool hasLeftNeighbor() const { return ( rankX > 0 ); }
	bool hasRightNeighbor() const { return ( rankX < ( processesX - 1 ) ); }
	bool hasTopNeighbor() const { return ( rankY > 0 ); }
	bool hasBottomNeighbor() const { return ( rankY < ( processesY - 1 ) ); }
	size_t rankByXY( size_t x, size_t y ) const
	{
		return ( y * processesX + x );
	}

	void setProcessXY();
	void setExchangeDefinitions();
	void allReduceFraction( CFraction& fraction );
	void allReduceDifference();
	void iteration0();
	void iteration1();
	void iteration2();
};

///////////////////////////////////////////////////////////////////////////////

void CProgram::Run( size_t pointsX, size_t pointsY, IIterationCallback& callback )
{
	CProgram program( pointsX, pointsY );

	// Выполняем нулевую итерацию (инициализацию).
	if( !callback.BeginIteration() ) {
		return;
	}
	program.iteration0();
	callback.EndIteration( program.difference );

	// Выполняем первую итерацию.
	if( !callback.BeginIteration() ) {
		return;
	}
	program.iteration1();
	callback.EndIteration( program.difference );

	// Выполняем остальные итерации.
	while( callback.BeginIteration() ) {
		program.iteration2();
		callback.EndIteration( program.difference );
	}
}

CProgram::CProgram( size_t pointsX, size_t pointsY ) :
	numberOfProcesses( CMpiSupport::NumberOfProccess() ),
	rank( CMpiSupport::Rank() ),
	pointsX( pointsX ), pointsY( pointsY ),
	difference( numeric_limits<NumericType>::max() )
{
	setProcessXY();
	rankX = rank % processesX;
	rankY = rank / processesX;
	GetBeginEndPoints( pointsX, processesX, rankX, beginX, endX );
	GetBeginEndPoints( pointsY, processesY, rankY, beginY, endY );

	if( hasLeftNeighbor() ) {
		beginX--;
	}
	if( hasRightNeighbor() ) {
		endX++;
	}
	if( hasTopNeighbor() ) {
		beginY--;
	}
	if( hasBottomNeighbor() ) {
		endY++;
	}

	// инициализируем grid
	grid.X.PartInit( -2.0, 2.0, pointsX, beginX, endX );
	grid.Y.PartInit( -2.0, 2.0, pointsY, beginY, endY );

	// заполняем список соседей с которыми будем обмениваться данными.
	setExchangeDefinitions();

#ifdef _DEBUG
	cout << "(" << rank << ")" << " {" << rankX << ", " << rankY << "}"
		<< " [" << beginX << ", " << endX << ")"
		<< " x [" << beginY << ", " << endY << ")" << endl;
	for( CExchangeDefinitions::const_iterator i = exchangeDefinitions.begin();
		i != exchangeDefinitions.end(); ++i ) {
		cout << rankX << " " << rankY << " "
			<< i->SendPart() << " " << i->RecvPart() << endl;
	}
#endif
}

void CProgram::setProcessXY()
{
	size_t power = 0;
	{
		size_t i = 1;
		while( i < numberOfProcesses ) {
			i *= 2;
			power++;
		}
		if( i != numberOfProcesses ) {
			throw CException( "The number of processes must be power of 2." );
		}
	}

	float pX = static_cast<float>( pointsX );
	float pY = static_cast<float>( pointsY );

	size_t powerX = 0;
	size_t powerY = 0;
	for( size_t i = 0; i < power; i++ ) {
		if( pX > pY ) {
			pX = pX / 2;
			powerX++;
		} else {
			pY = pY / 2;
			powerY++;
		}
	}

	processesX = 1 << powerX;
	processesY = 1 << powerY;
}

void CProgram::setExchangeDefinitions()
{
	if( hasLeftNeighbor() ) {
		exchangeDefinitions.push_back( CExchangeDefinition(
			rankByXY( rankX - 1, rankY ),
			grid.Column( 1, 1 /* decreaseTop */, 1 /* decreaseBottom */ ),
			grid.Column( 0, 1 /* decreaseTop */, 1 /* decreaseBottom */ ) ) );
	}
	if( hasRightNeighbor() ) {
		exchangeDefinitions.push_back( CExchangeDefinition(
			rankByXY( rankX + 1, rankY ),
			grid.Column( grid.X.Size() - 2, 1 /* decreaseTop */, 1 /* decreaseBottom */ ),
			grid.Column( grid.X.Size() - 1, 1 /* decreaseTop */, 1 /* decreaseBottom */ ) ) );
	}
	if( hasTopNeighbor() ) {
		exchangeDefinitions.push_back( CExchangeDefinition(
			rankByXY( rankX, rankY - 1 ),
			grid.Row( 1, 1 /* decreaseLeft */, 1 /* decreaseRight */ ),
			grid.Row( 0, 1 /* decreaseLeft */, 1 /* decreaseRight */ ) ) );
	}
	if( hasBottomNeighbor() ) {
		exchangeDefinitions.push_back( CExchangeDefinition(
			rankByXY( rankX, rankY + 1 ),
			grid.Row( grid.Y.Size() - 2, 1 /* decreaseLeft */, 1 /* decreaseRight */ ),
			grid.Row( grid.Y.Size() - 1, 1 /* decreaseLeft */, 1 /* decreaseRight */ ) ) );
	}
}

void CProgram::allReduceFraction( CFraction& fraction )
{
	NumericType buffer[2] = { fraction.Numerator, fraction.Denominator };
	MpiCheck( MPI_Allreduce( MPI_IN_PLACE, buffer, 2 /* count */,
		MpiNumericType, MPI_SUM, MPI_COMM_WORLD ), "MPI_Allreduce" );
	fraction.Numerator = buffer[0];
	fraction.Denominator = buffer[1];
}

void CProgram::allReduceDifference()
{
	MpiCheck( MPI_Allreduce( MPI_IN_PLACE, &difference, 1 /* count */,
		MpiNumericType, MPI_MAX, MPI_COMM_WORLD ), "MPI_Allreduce" );
}

void CProgram::iteration0()
{
	p.Init( grid.X.Size(), grid.Y.Size() );

	if( !hasLeftNeighbor() ) {
		for( size_t y = 0; y < p.SizeY(); y++ ) {
			p( 0, y ) = Phi( grid.X[0], grid.Y[y] );
		}
	}
	if( !hasRightNeighbor() ) {
		const size_t left = p.SizeX() - 1;
		for( size_t y = 0; y < p.SizeY(); y++ ) {
			p( left, y ) = Phi( grid.X[left], grid.Y[y] );
		}
	}
	if( !hasTopNeighbor() ) {
		for( size_t x = 0; x < p.SizeX(); x++ ) {
			p( x, 0 ) = Phi( grid.X[x], grid.Y[0] );
		}
	}
	if( !hasBottomNeighbor() ) {
		const size_t bottom = p.SizeY() - 1;
		for( size_t x = 0; x < p.SizeX(); x++ ) {
			p( x, bottom ) = Phi( grid.X[x], grid.Y[bottom] );
		}
	}
}

void CProgram::iteration1()
{
	r.Init( grid.X.Size(), grid.Y.Size() );

	CalcR( p, grid, r );
	exchangeDefinitions.Exchange( r );

	CFraction tau = CalcTau( r, r, grid );
	allReduceFraction( tau );

	difference = CalcP( r, tau.Value(), p );
	allReduceDifference();

	g = r;
}

void CProgram::iteration2()
{
	exchangeDefinitions.Exchange( p );

	CalcR( p, grid, r );
	exchangeDefinitions.Exchange( r );

	CFraction alpha = CalcAlpha( r, g, grid );
	allReduceFraction( alpha );

	CalcG( r, alpha.Value(), g );
	exchangeDefinitions.Exchange( g );

	CFraction tau = CalcTau( r, g, grid );
	allReduceFraction( tau );

	difference = CalcP( g, tau.Value(), p );
	allReduceDifference();
}

///////////////////////////////////////////////////////////////////////////////

// Последовательная реализация.
void Serial( const size_t pointsX, const size_t pointsY, IIterationCallback& callback )
{
	CUniformGrid grid;
	grid.X.Init( -2.0, 2.0, pointsX );
	grid.Y.Init( -2.0, 2.0, pointsY );

	CMatrix p( grid.X.Size(), grid.Y.Size() );
	CMatrix r( grid.X.Size(), grid.Y.Size() );

	NumericType difference = numeric_limits<NumericType>::max();

	// Выполняем нулевую итерацию (инициализацию).
	if( !callback.BeginIteration() ) {
		return;
	}
	for( size_t x = 0; x < p.SizeX(); x++ ) {
		p( x, 0 ) = Phi( grid.X[x], grid.Y[0] );
		p( x, p.SizeY() - 1 ) = Phi( grid.X[x], grid.Y[p.SizeY() - 1] );
	}
	for( size_t y = 1; y < p.SizeY() - 1; y++ ) {
		p( 0, y ) = Phi( grid.X[0], grid.Y[y] );
		p( p.SizeX() - 1, y ) = Phi( grid.X[p.SizeX() - 1], grid.Y[y] );
	}
	callback.EndIteration( difference );

	// Выполняем первую итерацию.
	if( !callback.BeginIteration() ) {
		return;
	}
	{
		CalcR( p, grid, r );
		const CFraction tau = CalcTau( r, r, grid );
		difference = CalcP( r, tau.Value(), p );
	}
	callback.EndIteration( difference );

	CMatrix g( r );
	// Выполняем остальные итерации.
	while( callback.BeginIteration() ) {
		CalcR( p, grid, r );
		const CFraction alpha = CalcAlpha( r, g, grid );
		CalcG( r, alpha.Value(), g );
		const CFraction tau = CalcTau( r, g, grid );
		difference = CalcP( g, tau.Value(), p );

		callback.EndIteration( difference );
	}
}

///////////////////////////////////////////////////////////////////////////////

void ParseArguments( const int argc, const char* const argv[],
	size_t& pointsX, size_t& pointsY )
{
	if( argc != 3 ) {
		throw CException( "too few arguments\nUsage: dirch POINTS_X POINTS_Y" );
	}

	pointsX = strtoul( argv[1], 0, 10 );
	pointsY = strtoul( argv[2], 0, 10 );

	if( pointsX == 0 || pointsY == 0 ) {
		throw CException( "invalid format of arguments\nUsage: dirch POINTS_X POINTS_Y" );
	}
}

void Main( const int argc, const char* const argv[] )
{
	size_t pointsX;
	size_t pointsY;
	ParseArguments( argc, argv, pointsX, pointsY );

	double programTime = 0.0;
	{
		auto_ptr<IIterationCallback> callback( new CSimpleIterationCallback );
		if( CMpiSupport::Rank() == 0 ) {
			callback.reset( new CIterationCallback( cout, 0 ) );
		}

		CMpiTimer timer( programTime );
		CProgram::Run( pointsX, pointsY, *callback );
	}
	cout << "(" << CMpiSupport::Rank() << ") Time: " << programTime << endl;
}

int main( int argc, char** argv )
{
	try {
		CMpiSupport::Initialize( &argc, &argv );
		Main( argc, argv );
		CMpiSupport::Finalize();
	} catch( exception& e ) {
		cerr << "Error: " << e.what() << endl;
		CMpiSupport::Abort( 1 );
		return 1;
	} catch( ... ) {
		cerr << "Unknown error!" << endl;
		CMpiSupport::Abort( 2 );
		return 2;
	}

	return 0;
}
