#include <Std.h>
#include <Definitions.h>
#include <MathFunctions.h>

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

// ���������� ������� rij �� ���������� ������.
void CalcR( const CMatrix&p, const CUniformGrid& grid, CMatrix& r )
{
	for( size_t x = 1; x < r.SizeX() - 1; x++ ) {
		for( size_t y = 1; y < r.SizeY() - 1; y++ ) {
			r( x, y ) = LaplasOperator( p, grid, x, y ) - F( grid.X[x], grid.Y[y] );
		}
	}
}

// ���������� �������� gij �� ���������� ������.
void CalcG( const CMatrix&r, const NumericType alpha, CMatrix& g )
{
	for( size_t x = 1; x < g.SizeX() - 1; x++ ) {
		for( size_t y = 1; y < g.SizeY() - 1; y++ ) {
			g( x, y ) = r( x, y ) - alpha * g( x, y );
		}
	}
}

// ���������� �������� pij �� ���������� ������, ������������ �������� �����.
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

// ���������� alpha.
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

// ���������� tau.
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