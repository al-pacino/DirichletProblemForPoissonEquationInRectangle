#pragma once

///////////////////////////////////////////////////////////////////////////////

NumericType LaplasOperator( const CMatrix& matrix, const CUniformGrid& grid, size_t x, size_t y );

// ���������� ������� rij �� ���������� ������.
void CalcR( const CMatrix&p, const CUniformGrid& grid, CMatrix& r );

// ���������� �������� gij �� ���������� ������.
void CalcG( const CMatrix&r, const NumericType alpha, CMatrix& g );

// ���������� �������� pij �� ���������� ������, ������������ �������� �����.
NumericType CalcP( const CMatrix&g, const NumericType tau, CMatrix& p );

// ���������� alpha.
CFraction CalcAlpha( const CMatrix&r, const CMatrix&g, const CUniformGrid& grid );

// ���������� tau.
CFraction CalcTau( const CMatrix&r, const CMatrix&g, const CUniformGrid& grid );

///////////////////////////////////////////////////////////////////////////////
