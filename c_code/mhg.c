/* MEX function ss=mhg(MAX,alpha,p,q,x,y)
 computes the truncated hypergeometric function pFq ^alpha(p;q;x;y)
 The sum is only over |kappa|<=MAX 
 p and q are arrays, so mhg(30,9,[3 4],[5 6 7],[0.5 0.6],[0.8,0.9]) is 
 2F3^9([3 4],[5 6 7];[0.5 0.6], [0.8 0.9]) summed over all kappa with 
 |kappa|<=30

 y may be omitted.

 Copyright, Plamen Koev, May 2004, April 2005


 Uses the C function hg(int MAX, double alpha, int n, double* x, 
                        double *y, int nx, double* p, double* q,
                        int np, int nq, double *s)
 which computes s = np F nq ^alpha(p,q,x,y) where the 
 summation is over |kappa|<=MAX and y may be NULL.

*/


#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

typedef struct {
     double alpha, *xn, *yn, *p, *q, s, *jx, *jy, *z;
     int n, np, nq, MAX, *D, *U, heap, *l, *lc, w, *m, *mc, wm;
} hg_glob;


void hg_jack1(int k, double aux, int sumlm, hg_glob* a)
{
   int  mi1,start,i,j,r,wc;
   double q0,q1,q2,aux1,t,dn,alpha1;

   if ((*a).lc[1]==1) start=2;
   else start=(*a).lc[1];

   if (k==0) i=1;
   else i=k;


   while (i<=(*a).mc[1]) {
      if (i<(*a).MAX) mi1=(*a).m[i]-(*a).m[i+1];
      else mi1=(*a).m[i];
      if (mi1>0) {
         t=(*a).l[i]-(*a).m[i];
         aux1=aux*(1+(*a).alpha*t);
         dn=t+1;
         alpha1=1-(*a).alpha; /* precomputing 1-alpha to save a subtraction 
                               in the loop below */

         t=i-(*a).alpha * (*a).m[i];
         for (r=1; r<i; r++) {
            q0=t-r;
            q1=q0+(*a).alpha * ((*a).l[r]+1);
            q2=q0+(*a).alpha * (*a).m[r];
            aux1*=(q1+alpha1)*(q2+(*a).alpha);
            dn*=q1*q2;
         }

         for (r=1;r<(*a).m[i];r++) {
            q1=(*a).mc[r]-t-(*a).alpha*r;
            aux1*=(q1+(*a).alpha);
            dn*=q1;
         }
         aux1/=dn;

         wc=(*a).wm;

         (*a).mc[(*a).m[i]]--;
         (*a).m[i]--;
         (*a).wm=(*a).U[(*a).wm*(*a).MAX+i-1];

         if ((*a).m[i]>0)
            hg_jack1(i,aux1,sumlm+1,a);
         else 
            if ((*a).jy==NULL)
               for (j=start-1;j<(*a).n;j++) 
                  (*a).jx[(*a).w*(*a).n + j]+=
                      (*a).jx[(*a).wm* (*a).n + j-1] * aux1 * 
                      (*a).xn[j*((*a).MAX+1) + sumlm+1];
            else
               for (j=start-1;j<(*a).n;j++) {
                  (*a).jx[(*a).w * (*a).n + j]+=
                      (*a).jx[(*a).wm * (*a).n+j-1] * aux1 * 
                      (*a).xn[j*((*a).MAX+1) + sumlm+1];
                  (*a).jy[(*a).w * (*a).n + j]+=
                      (*a).jy[(*a).wm * (*a).n+j-1] * aux1 * 
                      (*a).yn[j*((*a).MAX+1) + sumlm+1];
               }

         (*a).m[i]++;
         (*a).mc[(*a).m[i]]++;
         (*a).wm=wc;
      }
      i++;
   }

   if ((*a).jy==NULL)
      if (k==0)
         for (i=start-1;i<(*a).n;i++) (*a).jx[(*a).w * (*a).n+i]+=(*a).jx[(*a).w * (*a).n +i-1];
      else
         for (i=start-1;i<(*a).n;i++)
           (*a).jx[(*a).w * (*a).n+i]+=
              (*a).jx[(*a).wm*(*a).n + i-1] * aux * 
              (*a).xn[i*((*a).MAX+1)+sumlm];
   else
      if (k==0)
         for (i=start-1;i<(*a).n;i++) {
            (*a).jx[(*a).w*(*a).n+i]+=(*a).jx[(*a).w*(*a).n +i-1];
            (*a).jy[(*a).w*(*a).n+i]+=(*a).jy[(*a).w*(*a).n +i-1];
         }
      else
         for (i=start-1;i<(*a).n;i++){
           (*a).jx[(*a).w*(*a).n+i]+=
               (*a).jx[(*a).wm*(*a).n + i-1]*aux*(*a).xn[i*((*a).MAX+1)+sumlm];
           (*a).jy[(*a).w*(*a).n+i]+=
               (*a).jy[(*a).wm*(*a).n + i-1] * aux * 
               (*a).yn[i*((*a).MAX+1)+sumlm];
         }
}


void hg_summation(int i, int ms, hg_glob* a) {

  int j,m,ii,mm,lj1,lj2,jj, wold;
  double zn,dn,c,d,e,f,g;

  wold=(*a).w;
  m=ms;
  if (i>1) if ((*a).l[i-1]<m) m=(*a).l[i-1];
  for (ii=1;ii<m+1;ii++) {
     if ((ii==1)&&(i>1)) {
        (*a).D[(*a).w]=(*a).heap;
        (*a).w=(*a).heap;
        (*a).heap+=m;
     }
     else (*a).w++;

     (*a).l[i]=ii;
     (*a).lc[ii]++;        /* update conjugate partition */

     for (j=1;j<=(*a).lc[1];j++) {
        if (j<(*a).lc[1]) lj1=(*a).l[j+1];
        else lj1=0;
		 
        if ((*a).l[j]>lj1) {
	    mm=(*a).l[1];
	    if (j==1) mm--;
            for (jj=2;jj<=(*a).lc[1];jj++) {
	       if (jj==j) lj2=(*a).l[jj]-2;
               else lj2=(*a).l[jj]-1;
               if (lj2>=0) mm=(*a).D[mm]+lj2;
            }
	    (*a).U[(*a).w*(*a).MAX+j-1]=mm;
        }
     }

     dn=1;
     zn=1;
     c=-(i-1)/(*a).alpha+(*a).l[i]-1;
     for (j=0;j<(*a).np;j++)  zn*=(*a).p[j]+c;
     for (j=0;j<(*a).nq;j++)  dn*=(*a).q[j]+c;

     d=(*a).l[i]*(*a).alpha-i;             /* update j_lambda  */
     for (j=1;j<(*a).l[i];j++) {
        e=d-j*(*a).alpha+(*a).lc[j];
        g=e+1;
        zn*=(g-(*a).alpha)*e;
        dn*=g*(e+(*a).alpha);
     }
     for (j=1;j<i;j++) {
        f=(*a).l[j]*(*a).alpha-j-d;
        g=f+(*a).alpha;
        e=f*g;
        zn*=e-f;
        dn*=g+e;
     }
     (*a).z[i]*=zn/dn;

     if ((*a).lc[1]==1) {
        (*a).jx[(*a).w*(*a).n]=(*a).xn[1] * (*a).jx[((*a).w-1)*(*a).n] 
                 * (1+(*a).alpha*((*a).l[1]-1));
     if ((*a).jy!=NULL) 
        (*a).jy[(*a).w*(*a).n]=
           (*a).yn[1]*(*a).jy[((*a).w-1)*(*a).n]*(1+(*a).alpha*((*a).l[1]-1));
     }

     memcpy((*a).m,(*a).l,((*a).MAX+1)*sizeof(int));
     memcpy((*a).mc,(*a).lc,((*a).MAX+1)*sizeof(int));
     (*a).wm=(*a).w;
     hg_jack1(0,1,0,a);

     if ((*a).jy==NULL)  
        (*a).s += (*a).z[i]* (*a).jx[(*a).w*(*a).n+(*a).n-1];
     else {
        (*a).z[i]/=((*a).n+(*a).alpha*c);
        (*a).s += (*a).z[i] * (*a).jx[(*a).w*(*a).n+(*a).n-1] * 
                  (*a).jy[(*a).w*(*a).n+(*a).n-1];
     }
     if ((ms>ii)&&(i<(*a).n)) {
       (*a).z[i+1]=(*a).z[i];
       hg_summation(i+1,ms-ii,a);
     }
  }
  (*a).l[i]=0;
  for (ii=1; ii<m+1; ii++) (*a).lc[ii]--;
  (*a).w = wold;
}

double hg(int MAX, double alpha, int n, double* x, double*y, double* p, double* q,
                    int np, int nq) {
  int i,j,k,*f,ss,minn;

  //printf("HG!\n");
  //printf("%d\n",MAX);
  //printf("%f\n",alpha);
  //printf("%d\n",n);
  //printf("%f\n",x[0]);
  //printf("%lx\n",(size_t) y);
  //printf("%f\n",p[0]);
  //printf("%f\n",q[0]);
  //printf("%d\n",np);
  //printf("%d\n",nq);
  hg_glob *a;

  a=(hg_glob*)malloc(sizeof(hg_glob));
  (*a).n=n;
  (*a).MAX=MAX;
  (*a).alpha=alpha;
  (*a).p = p;
  (*a).q = q;
  (*a).np = np;
  (*a).nq = nq;
  (*a).s = 1;
  (*a).w = 0; /* index of the zero partition, currently l*/

  /* figure out the number of partitions |kappa|<= MAX with at most n parts */

  j=MAX+1;
  f=(int*) calloc (j*j,sizeof(int));
  for (i=1;i<j;i++) f[j+i]=1;
	
  ss=j;
  for (i=2;i<MAX+1;i++) {
     if (i+1<n+1) minn=i+1;
     else minn=n+1;
     for (k=2;k<minn;k++) {
        f[k*j+i]=f[(k-1)*j+i-1]+f[k*j+i-k];
        ss+=f[k*j+i];
     }
  }

  free(f);
  i=ss;

  (*a).jx=(double*) calloc(n*i,sizeof(double));
  if (y!=NULL) (*a).jy=(double*) calloc(n*i,sizeof(double));
  else (*a).jy=NULL;
  (*a).D=(int*) calloc(i,sizeof(int));
  (*a).U=(int*) calloc(MAX*i,sizeof(int));
  (*a).heap = MAX+1;


  (*a).xn=(double*) malloc(sizeof(double)*n*(MAX+1));
  for (i=0; i<n; i++) {
    (*a).jx[i]=1;
    (*a).xn[(MAX+1)*i]=1;
    for (j=1;j<MAX+1;j++) (*a).xn[(MAX+1)*i+j]=(*a).xn[(MAX+1)*i+j-1]*x[i];
  }

  if (y!=NULL) {
     (*a).yn=(double*) malloc(sizeof(double)*n*(MAX+1));
     for (i=0; i<n; i++) {
        (*a).jy[i]=1;
        (*a).yn[(MAX+1)*i]=1;
        for (j=1;j<MAX+1;j++) 
           (*a).yn[(MAX+1)*i+j]=(*a).yn[(MAX+1)*i+j-1]*y[i];
     }
  }

  (*a).z  =(double*)malloc((MAX+1) * sizeof(double));
  (*a).z[1]=1;
  (*a).l  =(int*)calloc(MAX+1,sizeof(int));
  (*a).lc =(int*)calloc(MAX+1,sizeof(int));
  (*a).m  =(int*)calloc(MAX+1,sizeof(int));
  (*a).mc =(int*)calloc(MAX+1,sizeof(int));

  hg_summation(1,MAX,a);

  free((*a).mc);
  free((*a).m);
  free((*a).lc);
  free((*a).l);
  free((*a).z);
  if (y!=NULL) free((*a).jy);
  free((*a).xn);
  free((*a).U);
  free((*a).D);
  free((*a).jx);

  double s=(*a).s;
  free(a);
  return s;

}

double x(double* y)
{
    printf("%f\n", y[0]);
    printf("%f\n", y[1]);
    double ret = y[0]+y[1];
    printf("%f\n", ret);

    return ret;
}
