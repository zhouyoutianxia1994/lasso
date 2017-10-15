#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
//'Coordinate descent method for solving Lasso
//'
//'@param y0 the response vector
//'@param X0 the design matrix
//'@param lambda the penalty parameter
//'@return estimator of beta
//'@example
//'require(coordinate_descent)
//'X=matrix(c(1,0.5,-1,-1),nrow=2)
//'y=c(2,-1)
//'lambda=0.1
//'coordinate_descent(X,y,lambda)

inline double soft_thresholding(double z, double lambda){
  double x;
  if(z > lambda)
    x = z-lambda;
  else if(z < (-lambda))
    x = z+lambda;
  else 
    x = 0;
  return  x;
}

inline arma::mat standardize_mat(const arma::mat& X){
  int p = X.n_cols;
  int N = X.n_rows;
  arma::mat S(N,p,fill::ones);
  arma::mat cent_X=X-S*diagmat(mean(X));
  arma::mat stand_X;
  arma::mat diag;
  diag=diagmat(pow(stddev(X,1,0),-1));
  stand_X=cent_X*diag;
  return stand_X;
}
// [[Rcpp::export]]

arma::vec  coordinate_descent(const arma::mat& X0, const arma::vec& y0, double lambda) {
  int p = X0.n_cols;
  int N = X0.n_rows;
  arma::vec s;
  s.ones(N);
  arma::vec y = y0-mean(y0)*s;
  arma::mat X = standardize_mat(X0);
  arma::vec beta0;
  beta0.zeros(p);
  arma::vec beta=beta0;
  arma::vec t=beta0;
  arma::vec r=y-X*beta0;
  double z;
  do{
  for(int k=0;k<p;k++){
  z=sum(r%X.col(k))/N+beta0(k);
  beta(k)=soft_thresholding(z, lambda);
  r=r-X.col(k)*(beta(k)-beta0(k));
  t(k)=beta0(k);
  beta0(k)=beta(k);
  }
  }
  while(sum((beta-t)%(beta-t))>0.01);
  beta=diagmat(pow(stddev(X0,1,0),-1))*beta;
  return beta;
}






