#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

//'proximal_operator method for solving Lasso
//'
//'@param y0 the response vector
//'@param X0 the design matrix
//'@param lambda the penalty parameter
//'@return estimator of beta
//'@example
//'require(proximal_operator)
//'X=matrix(c(1,0.5,-1,-1),nrow=2)
//'y=c(2,-1)
//'lambda=0.1
//'proximal_operator(X,y,lambda)

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

arma::vec proximal_operator(const  arma::mat X0, const arma::vec y0, double lambda) {
  int p = X0.n_cols;
  int N = X0.n_rows;
  arma::vec s;
  s.ones(N);
  arma::vec y = y0-mean(y0)*s;
  arma::mat X = standardize_mat(X0);
  arma::vec eigenvalue=eig_sym(X.t()*X);
  double M=max(eigenvalue)/N;
  arma::vec beta0(p,fill::zeros);
  arma::vec beta(p);
  arma::vec t;
  arma::vec z;
  do{
      t=beta0;
      z=beta0+(X.t()*(y-X*beta0))/(M*N);
      for(int k=0;k<p;k++){
      beta(k)=soft_thresholding(z(k), lambda/M);
      beta0(k)=beta(k);
      }
  }
  while(sum((beta-t)%(beta-t))>0.001);
  beta=diagmat(pow(stddev(X0,1,0),-1))*beta;
  return beta;
}
  
  
  
  
  