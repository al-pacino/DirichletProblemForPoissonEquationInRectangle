#include <Std.h>
#include <Definitions.h>
#include <MathObjects.h>
#include <Errors.h>

///////////////////////////////////////////////////////////////////////////////

CMatrix& CMatrix::operator=( const CMatrix& other )
{
	sizeX = other.sizeX;
	sizeY = other.sizeY;
	values = other.values;
	return *this;
}

void CMatrix::Init( const size_t _sizeX, const size_t _sizeY )
{
	sizeX = _sizeX;
	sizeY = _sizeY;
	values.resize( sizeX * sizeY );
	fill( values.begin(), values.end(), static_cast<NumericType>( 0 ) );
}

///////////////////////////////////////////////////////////////////////////////

ostream& operator<<( ostream& out, const CMatrixPart& matrixPart )
{
	out << "[" << matrixPart.BeginX << ", " << matrixPart.EndX << ") x "
		<< "[" << matrixPart.BeginY << ", " << matrixPart.EndY << ")";
	return out;
}

///////////////////////////////////////////////////////////////////////////////

void CUniformPartition::PartInit( NumericType p0, NumericType pN, size_t size, size_t begin, size_t end )
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

///////////////////////////////////////////////////////////////////////////////
