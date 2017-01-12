#ifndef sparfun_HPP
#define sparfun_HPP

//   R = SPRANDSYM(n,density) is a symmetric random, n-by-n, sparse
//       matrix with approximately density*n*n nonzeros; each entry is
//       the sum of one or more normally distributed random samples.
// http://opg1.ucsd.edu/~sio221/SIO_221A_2009/SIO_221_Data/Matlab5/Toolbox/matlab/sparfun/sprandsym.m
SMatrix sprandsym(int arg1, double density) {
  /*
  n = arg1;
  nl = round( (n*(n+1)/2) * density );
  rands = randn( nl, 1 );
  ii = fix( rand(nl, 1) * n ) + 1;
  jj = fix( rand(nl, 1) * n ) + 1;
  di = find( ii == jj );
  off = find( ii ~= jj );
  nd = length( di );
  no = length( off );
  randi = rands( 1:nd );
  rando = rands( nd+1 : nl );
  i = [ii(off); jj(off); ii(di)];
  j = [jj(off); ii(off); ii(di)];
  r = [rando; rando; randi];
  R = sparse(i,j,r,n,n);
  */
}

//   R = SPRAND(m,n,density) is a random, m-by-n, sparse matrix with
//       approximately density*m*n uniformly distributed nonzero entries.
//       SPRAND is designed to produce large matrices with small density
//       and will generate significantly fewer nonzeros than requested
//       if m*n is small or density is large.
// http://opg1.ucsd.edu/~sio221/SIO_221A_2009/SIO_221_Data/Matlab5/Toolbox/matlab/sparfun/sprand.m
SMatrix sprand(int arg1, int n, double density) {
  /*
  m = arg1;
  nnzwanted = round(m * n * min(density,1));
  i = fix( rand(nnzwanted, 1) * m ) + 1;
  j = fix( rand(nnzwanted, 1) * n ) + 1;
  ij = unique([i j],'rows');
  if ~isempty(ij)
     i = ij(:,1);
     j = ij(:,2);
  end
  rands = rand( length(i), 1 );
  R = sparse(i,j,rands,m,n);
  */
}

#endif
