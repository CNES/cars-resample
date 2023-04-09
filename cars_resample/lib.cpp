#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <array>
#include <list>
#include <cmath>
#include <algorithm>
#include <limits>

// -------------
// pure C++ code
// -------------

void nearestFiltering(const double accurateRowIn,
		      const double accurateColIn,
		      const long int nbRowsIn,
		      const long int nbColsIn,
		      const long int nbBands,
		      std::vector<float>& targetVector,
		      const long int kOut,
		      const long int sizeOut,
		      const std::vector<double>& sourceVector,
		      const long int sizeIn) {

  long int filterCenterRow = floor(accurateRowIn+0.5);
  long int filterCenterCol = floor(accurateColIn+0.5);
  
  if (filterCenterRow >= 0 && filterCenterRow < nbRowsIn && filterCenterCol >= 0 && filterCenterCol < nbColsIn) {
    long int kIn = filterCenterRow * nbColsIn + filterCenterCol;
    for ( long int b = 0; b < nbBands ; ++b) {
      targetVector[kOut+b*sizeOut] = sourceVector[kIn + b*sizeIn];
    }
  }
}

std::vector<double> computeBicubicFilterWeights(const double relativeCoord) {
  // Compute bicubic filter weights
  // w[0]: -2, -1[, w[1]: [-1, 0[, w[2]: [0, 1[, w[3]: [1, 2[
  // w = (a+2)|x|**3 -(a+3)|x|**2 + 1  if |x| < 1
  //   = a|x|**3 -5a|x|**2 + 8a|x| -4a if 1 < |x| < 2
  // with a = -0.5

  std::vector<double> weights(5);
  double x;
  for (long int k=-2; k<=2; ++k) {
    x = abs(relativeCoord + k);
    if (x < 1) {
      weights[k+2] = x * x * (1.5 * x - 2.5) + 1;    
    }
    else if (x < 2) {
      weights[k+2] = x * (x * (-0.5 * x + 2.5) - 4) + 2;
    }
    else {
      weights[k+2] = 0;
    }
  }

  return weights;
}

void bicubicFiltering(const double accurateRowIn,
		      const double accurateColIn,
		      const long int nbRowsIn,
		      const long int nbColsIn,
		      const long int nbBands,
		      std::vector<float>& targetVector,
		      const long int kOut,
		      const long int sizeOut,
		      const std::vector<double>& sourceVector,
		      const long int sizeIn) {

  long int filterCenterRow = floor(accurateRowIn+0.5);
  long int filterCenterCol = floor(accurateColIn+0.5);
  long int rowIn, colIn;

  if (filterCenterRow >= 0 && filterCenterRow < nbRowsIn && filterCenterCol >= 0 && filterCenterCol < nbColsIn) {
    long int kIn;
    double relativeRow, relativeCol;

    // relative greater or equal to 0 and lesser than 1
    relativeRow = accurateRowIn - filterCenterRow;
    relativeCol = accurateColIn - filterCenterCol;
    
    auto weightsRow = computeBicubicFilterWeights(relativeRow);
    auto weightsCol = computeBicubicFilterWeights(relativeCol);
    
    std::vector<double> interpCol(5*nbBands, 0.0);
    
    // interpolation along row direction : input 5x5 => output 5x1
    for ( int neighRowIn = -2; neighRowIn <= 2; ++neighRowIn ) {
      rowIn = filterCenterRow + neighRowIn;
      if (rowIn < 0) {
	// mirror: rowIn = -rowIn + 1;
	rowIn = 0;
      }
      else if (rowIn >= nbRowsIn){
	// mirror: rowIn = nbRowsIn - 1 + (nbRowsIn - rowIn) - 1;
	rowIn = nbRowsIn - 1;
      }
      
      for ( int neighColIn = -2; neighColIn <= 2; ++neighColIn ) {
	colIn = filterCenterCol + neighColIn;
	
	if (colIn < 0) {
	  // mirror: colIn = -colIn + 1;
	  colIn = 0;
	}
	else if (colIn >= nbColsIn){
	  // mirror: colIn = nbColsIn - 1 + (nbColsIn - colIn) - 1;
	  colIn = nbColsIn-1;
	}
	
	kIn = rowIn * nbColsIn + colIn;
	for ( long int b = 0; b < nbBands ; ++b) {    
	  interpCol[(neighRowIn+2)+b*5] += weightsCol[2-neighColIn]*sourceVector[kIn + b*sizeIn]; 
	}
      }
    }
    
    // interpolation along col direction : input 5x1 => output 1x1
    for ( long int b = 0; b < nbBands ; ++b) {
      double targetValue = 0;
      for ( int neighRowIn = -2; neighRowIn <= 2; ++neighRowIn ) {
      	targetValue += weightsRow[2-neighRowIn]*interpCol[(neighRowIn+2)+b*5];
      }
      targetVector[kOut+b*sizeOut] = targetValue;
    }
  }
}


std::vector<float> gridResampling(const std::vector<double>& sourceVector,
				  const std::vector<double>& gridVector,
				  const long int nbRowsIn,
				  const long int nbColsIn,
				  const long int nbRowsGrid,
				  const long int nbColsGrid,
				  const long int nbBands,
				  const long int oversampling,
				  const std::string interpolator,
				  const double nodata)
{
  const long int nbRowsOut = (nbRowsGrid-1)*oversampling+1;
  const long int nbColsOut = (nbColsGrid-1)*oversampling+1;

  const long int sizeIn = nbColsIn*nbRowsIn;
  const long int sizeGrid = nbColsGrid*nbRowsGrid;
  const long int sizeOut = nbColsOut*nbRowsOut;
  std::vector<float> targetVector(nbBands*sizeOut, nodata);

  double accurateRowIn, accurateColIn;

  long int rowGrid, colGrid;
  long int colAlpha, rowAlpha;
  long int kGrid1, kGrid2, kGrid3, kGrid4;
  long int rowOut, colOut;

  void (*filtering)(const double,
		    const double,
		    const long int,
		    const long int,
		    const long int,
		    std::vector<float>&,
		    const long int,
		    const long int,
		    const std::vector<double>&,
		    const long int);

  if (interpolator == "bicubic") {
    filtering = &bicubicFiltering;
  }
  else {
    filtering = &nearestFiltering;
  }

  for ( long int kOut = 0 ; kOut < sizeOut ; ++kOut ) {

    // 1. bilinear grid interpolation with oversampling

    // retrieve grid coordinates
    colOut = kOut % nbColsOut;
    rowOut = kOut / nbColsOut;

    colGrid = colOut / oversampling;
    rowGrid = rowOut / oversampling;

    // get 4 involved pixels
    kGrid1 = colGrid + rowGrid * nbColsGrid;
    kGrid2 = (colGrid+1) + rowGrid * nbColsGrid;
    kGrid3 = (colGrid+1) + (rowGrid+1) * nbColsGrid;
    kGrid4 = colGrid + (rowGrid+1) * nbColsGrid;

    // alpha factor to weight pixels
    colAlpha = colOut % oversampling;
    rowAlpha = rowOut % oversampling;

    // get column and row in source image
    accurateColIn = gridVector[kGrid1] * (oversampling-colAlpha) * (oversampling-rowAlpha);
    accurateColIn += gridVector[kGrid2] * colAlpha * (oversampling-rowAlpha);
    accurateColIn += gridVector[kGrid3] * colAlpha * rowAlpha;
    accurateColIn += gridVector[kGrid4] * (oversampling-colAlpha) * rowAlpha;
    accurateColIn /= oversampling*oversampling;

    accurateRowIn = gridVector[sizeGrid+kGrid1] * (oversampling-colAlpha) * (oversampling-rowAlpha);
    accurateRowIn += gridVector[sizeGrid+kGrid2] * colAlpha * (oversampling-rowAlpha);
    accurateRowIn += gridVector[sizeGrid+kGrid3] * colAlpha * rowAlpha;
    accurateRowIn += gridVector[sizeGrid+kGrid4] * (oversampling-colAlpha) * rowAlpha;
    accurateRowIn /= oversampling*oversampling;

    // filter center
    accurateRowIn -= 0.5;
    accurateColIn -= 0.5;

    // 2. filtering (nearest or bicubic)
    filtering(accurateRowIn, accurateColIn, nbRowsIn, nbColsIn,
	      nbBands, targetVector, kOut, sizeOut, sourceVector, sizeIn);
  }
  
  return targetVector;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::array pyGrid(py::array_t<double,
		 py::array::c_style | py::array::forcecast> source,
		 py::array_t<double,
		 py::array::c_style | py::array::forcecast> grid,
		 size_t oversampling,
		 std::string interpolator,
		 double nodata)
{
  // check input dimensions
  if ( source.ndim()     != 3 )
    throw std::runtime_error("source should be 3-D NumPy array");
  size_t nbBands = source.shape()[0];
  size_t nbRowsIn = source.shape()[1];
  size_t nbColsIn = source.shape()[2];

  if ( grid.ndim()     != 3 )
    throw std::runtime_error("grid should be 3-D NumPy array");

  if ( grid.shape()[0] != 2 )
    throw std::runtime_error("grid should have size [2, NROWS, NCOLS]");

  size_t nbRowsGrid = grid.shape()[1];
  size_t nbColsGrid = grid.shape()[2];
  size_t nbRowsOut = (nbRowsGrid-1)*oversampling+1;
  size_t nbColsOut = (nbColsGrid-1)*oversampling+1;

  if ((interpolator != "nearest") && (interpolator != "bicubic")) {
    throw std::runtime_error("interpolator not available");
  }

  // allocate std::vector (to pass to the C++ function)
  std::vector<double> sourceVector(source.size());
  std::vector<double> gridVector(grid.size());

  // copy py::array -> std::vector
  // Do we need to make a copy here ? Is there something similar
  // to cython memory view
  // https://stackoverflow.com/questions/54793539/pybind11-modify-numpy-array-from-c
  // It seems that passing the py::array by reference should do the job if
  // there is not conflicting type, however the function pointCloudToDSM needs a double * ptr as
  // input instead of a std::vector.
  std::memcpy(sourceVector.data(),source.data(),source.size()*sizeof(double));
  std::memcpy(gridVector.data(),grid.data(),grid.size()*sizeof(double));

  // call pure C++ function
  auto tgtVector = gridResampling(sourceVector,
				  gridVector,
				  nbRowsIn,
				  nbColsIn,
				  nbRowsGrid,
				  nbColsGrid,
				  nbBands,
				  oversampling,
				  interpolator,
				  nodata);

  size_t              ndim    = 3;
  std::vector<size_t> shape   = { nbBands, nbRowsOut, nbColsOut };
  std::vector<size_t> strides = { nbRowsOut*nbColsOut*sizeof(float),
				  nbColsOut*sizeof(float),
				  sizeof(float)};
  
  // return 2-D NumPy array, I think here it is ok since the expected argument is
  // a pointer so there is no copy
  return py::array(py::buffer_info(tgtVector.data(),
				   sizeof(float), 
				   py::format_descriptor<float>::format(),
				   ndim,
				   shape,
				   strides
				   ));
}

// wrap as Python module
PYBIND11_MODULE(resample, m)
{
  m.doc() = "resample";

  m.def("grid", &pyGrid, "Resampled a source image to a target image according a resampling grid",
	py::arg("source"),
	py::arg("grid"),
	py::arg("oversampling"),
	py::arg("interpolator"),
	py::arg("nodata"));
}
