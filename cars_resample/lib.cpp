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

inline void nearestFiltering(const double accurateRowIn,
		      const double accurateColIn,
		      const long int nbRowsIn,
		      const long int nbColsIn,
		      const long int nbBands,
		      std::vector<float>& targetVector,
		      const long int kOut,
		      const long int sizeOut,
		      const std::vector<double>& sourceVector,
		      const long int sizeIn) {

  long int filterCenterRow = static_cast<long int>(std::floor(accurateRowIn + 0.5));
  long int filterCenterCol = static_cast<long int>(std::floor(accurateColIn + 0.5));
  
  if (filterCenterRow >= 0 && filterCenterRow < nbRowsIn && filterCenterCol >= 0 && filterCenterCol < nbColsIn) {
    long int kIn = filterCenterRow * nbColsIn + filterCenterCol;
    for ( long int b = 0; b < nbBands ; ++b) {
      targetVector[kOut+b*sizeOut] = sourceVector[kIn + b*sizeIn];
    }
  }
}

inline double bicubicFilterWeightsCalcul1(double x) {
  return x * x * (1.5 * x - 2.5) + 1;
}
inline double bicubicFilterWeightsCalcul2(double x) {
  return x * (x * (-0.5 * x + 2.5) - 4) + 2;
}

inline std::vector<double> computeBicubicFilterWeights(const double relativeCoord) {
  // Compute bicubic filter weights
  // w[0]: -2, -1[, w[1]: [-1, 0[, w[2]: [0, 1[, w[3]: [1, 2[
  // w = (a+2)|x|**3 -(a+3)|x|**2 + 1  if |x| < 1
  //   = a|x|**3 -5a|x|**2 + 8a|x| -4a if 1 < |x| < 2
  // with a = -0.5

  double x;
  std::vector<double> weights(5);
  // WDL unrool for loop : Normaly values are only between -0.5 & +0.5
  if (relativeCoord < 0 && relativeCoord > -1) {
    weights[0] = 0;
    // - instead of abs because we know relativeCoord is negative
    weights[1] = bicubicFilterWeightsCalcul2(-relativeCoord + 1);
    weights[2] = bicubicFilterWeightsCalcul1(-relativeCoord);
    weights[3] = bicubicFilterWeightsCalcul1(relativeCoord + 1);
    weights[4] = bicubicFilterWeightsCalcul2(relativeCoord + 2);
  } else if (relativeCoord == 0) {
    weights[0] = 0;
    weights[1] = bicubicFilterWeightsCalcul2(1);
    weights[2] = 0;
    weights[3] = bicubicFilterWeightsCalcul2(1);
    weights[4] = 0;
  } else if (relativeCoord > 0 && relativeCoord < 1) {
    // - instead of abs because we know relativeCoord is negative
    weights[0] = bicubicFilterWeightsCalcul2(-relativeCoord + 2);
    weights[1] = bicubicFilterWeightsCalcul1(-relativeCoord + 1);
    weights[2] = bicubicFilterWeightsCalcul1(relativeCoord);
    weights[3] = bicubicFilterWeightsCalcul2(relativeCoord + 1);
    weights[4] = 0;
  } else { // keep old code in the other case (non regression of fct)
    for (int k=-2; k<=2; ++k) {
      x = abs(relativeCoord + k);
      if (x < 1) {
        weights[k+2] = bicubicFilterWeightsCalcul1(x);    
      }
      else if (x < 2) {
        weights[k+2] = bicubicFilterWeightsCalcul2(x);
      }
      else {
        weights[k+2] = 0;
      }
    }
  }

  return weights;
}

inline void bicubicFiltering(const double accurateRowIn,
		      const double accurateColIn,
		      const long int nbRowsIn,
		      const long int nbColsIn,
		      const long int nbBands,
		      std::vector<float>& targetVector,
		      const long int kOut,
		      const long int sizeOut,
		      const std::vector<double>& sourceVector,
		      const long int sizeIn) {

  long int filterCenterRow = static_cast<long int>(std::floor(accurateRowIn + 0.5));
  long int filterCenterCol = static_cast<long int>(std::floor(accurateColIn + 0.5));
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
    for (long int b = 0; b < nbBands; ++b) {
      double outputValue = 0.;
      for ( int neighRowIn = -2; neighRowIn <= 2; ++neighRowIn ) {
        rowIn = filterCenterRow + neighRowIn;

        // mirror: rowIn = -rowIn + 1;
        rowIn = (rowIn < 0) ? 0 : rowIn;
        // mirror: rowIn = nbRowsIn - 1 + (nbRowsIn - rowIn) - 1;
        rowIn = (rowIn >= nbRowsIn) ? nbRowsIn - 1 : rowIn;

        for ( int neighColIn = -2; neighColIn <= 2; ++neighColIn ) {
  	  colIn = filterCenterCol + neighColIn;

          // mirror: colIn = -colIn + 1;
          colIn = (colIn < 0) ? 0 : colIn;
          // mirror: colIn = nbColsIn - 1 + (nbColsIn - colIn) - 1;
          colIn = (colIn >= nbColsIn) ? nbColsIn - 1 : colIn;

          kIn = rowIn * nbColsIn + colIn;
          interpCol[(neighRowIn + 2) + b * 5] +=
              weightsCol[2 - neighColIn] * sourceVector[kIn + b * sizeIn];
        }

        // interpolation along col direction : input 5x1 => output 1x1
        outputValue += weightsRow[2 - neighRowIn] * interpCol[(neighRowIn + 2) + b * 5];
      }
      targetVector[kOut + b * sizeOut] = outputValue;
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

  long int rowGrid = 0, colGrid = 0;
  long int colAlpha = 0, rowAlpha = 0;
  long int kGrid1 = 0, kGrid2 = 1, kGrid3 = nbColsGrid + 1, kGrid4 = nbColsGrid;
  long int rowOut = 0, colOut = 0;
  double osP2 = (double)(oversampling * oversampling);
  long os_colA = (oversampling - colAlpha);
  long os_rowA = (oversampling - rowAlpha);


  bool bc = (interpolator == "bicubic");

  for ( long int kOut = 0 ; kOut < sizeOut  ; ++kOut ) {

    // 1. bilinear grid interpolation with oversampling
    // get column and row in source image
    double coef1 = (double)(os_colA * os_rowA);
    double coef2 = (double)(colAlpha * os_rowA);
    double coef3 = (double)(colAlpha * rowAlpha);
    double coef4 = (double)(os_colA * rowAlpha);
    // printf("\tos_calA:%ld, os_rowA:%ld\n", os_colA, os_rowA); // WDL DEBUG PRINT
    // printf("\tCoef %.0lf, %.0lf, %.0lf, %.0lf,\n", coef1, coef2, coef3, coef4); // WDL DEBUG PRINT

    accurateColIn = gridVector[kGrid1] * coef1;
    accurateColIn += gridVector[kGrid2] * coef2;
    accurateColIn += gridVector[kGrid3] * coef3;
    accurateColIn += gridVector[kGrid4] * coef4;
    accurateColIn /= osP2;

    accurateRowIn = gridVector[sizeGrid+kGrid1] * coef1;
    accurateRowIn += gridVector[sizeGrid+kGrid2] * coef2;
    accurateRowIn += gridVector[sizeGrid+kGrid3] * coef3;
    accurateRowIn += gridVector[sizeGrid+kGrid4] * coef4;
    accurateRowIn /= osP2;

    // filter center
    accurateRowIn -= 0.5;
    accurateColIn -= 0.5;

    // 2. filtering (nearest or bicubic)
     if (bc) {
    bicubicFiltering(accurateRowIn, accurateColIn, nbRowsIn, nbColsIn,
	      nbBands, targetVector, kOut, sizeOut, sourceVector, sizeIn);
     }else{
    nearestFiltering(accurateRowIn, accurateColIn, nbRowsIn, nbColsIn,
	      nbBands, targetVector, kOut, sizeOut, sourceVector, sizeIn);
     }
    /**3. Increment for next call :
      - grid coordinates
      - 4 involved pixel
      - alpha factor to weight pixels
      */
    colAlpha++;
    if (colAlpha == oversampling) {
      colAlpha = 0;
      colGrid++;
      kGrid1++;
      kGrid2++;
      kGrid3++;
      kGrid4++;
    }
    colOut++;
    if (colOut == nbColsOut) {
      colOut = 0;
      rowOut++;
      colGrid = 0;
      colAlpha = 0;
      rowAlpha++;
      if (rowAlpha == oversampling) {
        rowAlpha = 0;
        rowGrid++;
      }
      kGrid1 = colGrid + rowGrid * nbColsGrid;
      kGrid2 = kGrid1 + 1;
      kGrid4 = kGrid1 + nbColsGrid;
      kGrid3 = kGrid4 + 1;
      os_rowA = (oversampling - rowAlpha);
    }
    os_colA = (oversampling - colAlpha);
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
