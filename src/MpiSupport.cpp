#include <Std.h>
#include <Errors.h>
#include <MpiSupport.h>

///////////////////////////////////////////////////////////////////////////////

string CMpiException::makeErrorText() const
{
	ostringstream oss;
	oss << "MPI function '" << FunctionName() << "'"
		<< " has failed with code '" << ErrorCode() << "'.";
	return oss.str();
}

///////////////////////////////////////////////////////////////////////////////

void MpiCheck( const int mpiResult, const string& mpiFunctionName )
{
	if( mpiResult != MPI_SUCCESS ) {
		throw CMpiException( mpiResult, mpiFunctionName );
	}
}

///////////////////////////////////////////////////////////////////////////////

bool CMpiSupport::initialized = false;
size_t CMpiSupport::rank = 0;
size_t CMpiSupport::numberOfProccess = 0;

void CMpiSupport::Initialize( int* argc, char*** argv )
{
	if( Initialized() ) {
		throw CException( "MPI was already initialized!" );
	}
	MpiCheck( MPI_Init( argc, argv ), "MPI_Init" );
	int tmp;
	MpiCheck( MPI_Comm_rank( MPI_COMM_WORLD, &tmp ), "MPI_Comm_rank" );
	rank = static_cast<size_t>( tmp );
	MpiCheck( MPI_Comm_size( MPI_COMM_WORLD, &tmp ), "MPI_Comm_size" );
	numberOfProccess = static_cast<size_t>( tmp );
	initialized = true;
}

void CMpiSupport::Finalize()
{
	checkInitialized();
	MPI_Finalize();
}

void CMpiSupport::Abort( int code )
{
	if( Initialized() ) {
		MPI_Abort( MPI_COMM_WORLD, code );
	}
}

void CMpiSupport::checkInitialized()
{
	if( !Initialized() ) {
		throw CException( "MPI was not initialized yet!" );
	}
}

void CMpiSupport::Barrier()
{
	MpiCheck( MPI_Barrier( MPI_COMM_WORLD ), "MPI_Barrier" );
}

///////////////////////////////////////////////////////////////////////////////
