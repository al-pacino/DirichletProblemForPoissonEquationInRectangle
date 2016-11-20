#pragma once

///////////////////////////////////////////////////////////////////////////////

NumericType LaplasOperator( const CMatrix& matrix, const CUniformGrid& grid, size_t x, size_t y );

// Вычисление невязки rij во внутренних точках.
void CalcR( const CMatrix&p, const CUniformGrid& grid, CMatrix& r );

// Вычисление значений gij во внутренних точках.
void CalcG( const CMatrix&r, const NumericType alpha, CMatrix& g );

// Вычисление значений pij во внутренних точках, возвращается максимум норма.
NumericType CalcP( const CMatrix&g, const NumericType tau, CMatrix& p );

// Вычисление alpha.
CFraction CalcAlpha( const CMatrix&r, const CMatrix&g, const CUniformGrid& grid );

// Вычисление tau.
CFraction CalcTau( const CMatrix&r, const CMatrix&g, const CUniformGrid& grid );

///////////////////////////////////////////////////////////////////////////////
