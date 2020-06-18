#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>


typedef struct {
     double alpha, *xn, *yn, *p, *q, s, *jx, *jy, *z;
     int n, np, nq, MAX, *D, *U, heap, *l, *lc, w, *m, *mc, wm;
} loghg_glob;


void loghg_jack1(int k, double aux, int sumlm, loghg_glob* a) {

int  mi1,start,i,j,r,wc;
double q0,q1,q2,aux1,t,dn;

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
        t=i-(*a).alpha * (*a).m[i];
        for (r=1; r<i; r++) {
           q0=t-r;
           q1=q0+(*a).alpha * ((*a).l[r]+1);
           q2=q0+(*a).alpha * (*a).m[r];
           aux1*=(q1+1-(*a).alpha)*(q2+(*a).alpha);
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
           loghg_jack1(i,aux1,sumlm+1,a);
        else 
           if ((*a).jy==NULL)
              for (j=start-1;j<(*a).n;j++)     /* LOGGGG */
				  if ((*a).jx[(*a).w*(*a).n + j]==-exp(1000))
   				      (*a).jx[(*a).w*(*a).n + j]=(*a).jx[(*a).wm* (*a).n + j-1] +log(aux1) + (*a).xn[j*((*a).MAX+1) + sumlm+1];
				  else
  				      (*a).jx[(*a).w*(*a).n + j]+=
 				         log(1+exp((*a).jx[(*a).wm* (*a).n + j-1]-(*a).jx[(*a).w*(*a).n + j]) 
						     * aux1 * exp((*a).xn[j*((*a).MAX+1) + sumlm+1]));
		   else
			   for (j=start-1;j<(*a).n;j++) {
   				  if ((*a).jx[(*a).w*(*a).n + j]==-exp(1000))
   				      (*a).jx[(*a).w*(*a).n + j]=(*a).jx[(*a).wm* (*a).n + j-1] +log(aux1) + (*a).xn[j*((*a).MAX+1) + sumlm+1];
				  else
  				      (*a).jx[(*a).w*(*a).n + j]+=
 				         log(1+exp((*a).jx[(*a).wm* (*a).n + j-1]-(*a).jx[(*a).w*(*a).n + j]) 
						     * aux1 * exp((*a).xn[j*((*a).MAX+1) + sumlm+1]));

				  if ((*a).jy[(*a).w*(*a).n + j]==-exp(1000))
   				      (*a).jy[(*a).w*(*a).n + j]=(*a).jy[(*a).wm* (*a).n + j-1] +log(aux1) + (*a).yn[j*((*a).MAX+1) + sumlm+1];
				  else
  				      (*a).jy[(*a).w*(*a).n + j]+=
 				         log(1+exp((*a).jy[(*a).wm* (*a).n + j-1]-(*a).jy[(*a).w*(*a).n + j]) 
						     * aux1 * exp((*a).yn[j*((*a).MAX+1) + sumlm+1]));
			   }

        (*a).m[i]++;
        (*a).mc[(*a).m[i]]++;
		(*a).wm=wc;
      }
      i++;
   }

   if ((*a).jy==NULL)
      if (k==0)
         for (i=start-1;i<(*a).n;i++)   /* LOGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG */
			 if ((*a).jx[(*a).w * (*a).n+i]==-exp(1000))
                (*a).jx[(*a).w * (*a).n+i]=(*a).jx[(*a).w * (*a).n +i-1];
			 else
  			    (*a).jx[(*a).w * (*a).n+i]+=
			       log(1+exp((*a).jx[(*a).w * (*a).n +i-1]-(*a).jx[(*a).w * (*a).n+i]));
      else
         for (i=start-1;i<(*a).n;i++)
			 if ((*a).jx[(*a).w * (*a).n+i]==-exp(1000))
                (*a).jx[(*a).w * (*a).n+i]=(*a).jx[(*a).wm*(*a).n + i-1] + log(aux) + (*a).xn[i*((*a).MAX+1)+sumlm];
			 else
                (*a).jx[(*a).w * (*a).n+i]+= log(1+
		             exp((*a).jx[(*a).wm*(*a).n + i-1]-(*a).jx[(*a).w * (*a).n+i]) 
					    * aux * exp((*a).xn[i*((*a).MAX+1)+sumlm]));
   else
      if (k==0)
		  for (i=start-1;i<(*a).n;i++) {  /* LOGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG */
			 if ((*a).jx[(*a).w * (*a).n+i]==-exp(1000))
                (*a).jx[(*a).w * (*a).n+i]=(*a).jx[(*a).w * (*a).n +i-1];
			 else
  			    (*a).jx[(*a).w * (*a).n+i]+=
			       log(1+exp((*a).jx[(*a).w * (*a).n +i-1]-(*a).jx[(*a).w * (*a).n+i]));

			 if ((*a).jy[(*a).w * (*a).n+i]==-exp(1000))
                (*a).jy[(*a).w * (*a).n+i]=(*a).jy[(*a).w * (*a).n +i-1];
			 else
  			    (*a).jy[(*a).w * (*a).n+i]+=
			       log(1+exp((*a).jy[(*a).w * (*a).n +i-1]-(*a).jy[(*a).w * (*a).n+i]));
		  }
      else
		  for (i=start-1;i<(*a).n;i++) {
			 if ((*a).jx[(*a).w * (*a).n+i]==-exp(1000))
                (*a).jx[(*a).w * (*a).n+i]=(*a).jx[(*a).wm*(*a).n + i-1] + log(aux) + (*a).xn[i*((*a).MAX+1)+sumlm];
			 else
                (*a).jx[(*a).w * (*a).n+i]+= log(1+
		             exp((*a).jx[(*a).wm*(*a).n + i-1]-(*a).jx[(*a).w * (*a).n+i]) 
					    * aux * exp((*a).xn[i*((*a).MAX+1)+sumlm]));

			 if ((*a).jy[(*a).w * (*a).n+i]==-exp(1000))
                (*a).jy[(*a).w * (*a).n+i]=(*a).jy[(*a).wm*(*a).n + i-1] + log(aux) + (*a).yn[i*((*a).MAX+1)+sumlm];
			 else
                (*a).jy[(*a).w * (*a).n+i]+= log(1+
		             exp((*a).jy[(*a).wm*(*a).n + i-1]-(*a).jy[(*a).w * (*a).n+i]) 
					    * aux * exp((*a).yn[i*((*a).MAX+1)+sumlm]));
		  }
}


void loghg_summation(int i, int ms, loghg_glob* a) {
    printf("summation\n");
    printf("%d\n", i);
    printf("%d\n", ms);

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

    printf("a\n");
     for (j=1;j<=(*a).lc[1];j++) {
		 if (j<(*a).lc[1]) lj1=(*a).l[j+1];
		 else lj1=0;
		 
		 if ((*a).l[j]>lj1) {
			 mm=(*a).l[1];
			 if (j==1) mm--;
             printf("%d\n",(*a).lc[1]);
			 for (jj=2;jj<=(*a).lc[1];jj++) {
				 if (jj==j) lj2=(*a).l[jj]-2;
				 else lj2=(*a).l[jj]-1;
				 if (lj2>=0) mm=(*a).D[mm]+lj2;
			 }
			 (*a).U[(*a).w*(*a).MAX+j-1]=mm;
		 }
	 }

    printf("b\n");
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
		 (*a).jx[(*a).w*(*a).n]=
			 (*a).xn[1] + (*a).jx[((*a).w-1)*(*a).n] + log(1+(*a).alpha*((*a).l[1]-1)); /* LOGGGGG */
		 if ((*a).jy!=NULL) 
		 (*a).jy[(*a).w*(*a).n]=
			 (*a).yn[1] + (*a).jy[((*a).w-1)*(*a).n] + log(1+(*a).alpha*((*a).l[1]-1));
	 }

    printf("c\n");
	 memcpy((*a).m,(*a).l,((*a).MAX+1)*sizeof(int));
	 memcpy((*a).mc,(*a).lc,((*a).MAX+1)*sizeof(int));
	 (*a).wm=(*a).w;
	 loghg_jack1(0,1,0,a);

     if ((*a).jy==NULL)  
		 (*a).s +=log(1+(*a).z[i]* exp((*a).jx[(*a).w*(*a).n+(*a).n-1]-(*a).s)); /* LOGGGGGGGGGGGGGGGG */
	 else {
		 (*a).z[i]/=((*a).n+(*a).alpha*c);
		 (*a).s +=log(1+(*a).z[i] * exp( (*a).jx[(*a).w*(*a).n+(*a).n-1] +(*a).jy[(*a).w*(*a).n+(*a).n-1] - (*a).s));
	 }
    printf("d\n");
	 if ((ms>ii)&&(i<(*a).n)) {
		 (*a).z[i+1]=(*a).z[i];
	     loghg_summation(i+1,ms-ii,a);
	 }
  }
  (*a).l[i]=0;
  for (ii=1; ii<m+1; ii++) (*a).lc[ii]--;
  (*a).w = wold;
}



void loghg(int MAX, double alpha, int n, double* x, double*y, double* p, double* q,
                    int np, int nq, double *s) {
  int i,j,k,*f,ss,minn;
  loghg_glob *a;

  printf("logHG!\n");
  printf("%d\n",MAX);
  printf("%f\n",alpha);
  printf("%d\n",n);
  printf("%f\n",x[0]);
  printf("%lx\n",(size_t) y);
  printf("%f\n",p[0]);
  printf("%f\n",q[0]);
  printf("%d\n",np);
  printf("%d\n",nq);


  a=(loghg_glob*)malloc(sizeof(loghg_glob));
  (*a).n=n;
  (*a).MAX=MAX;
  (*a).alpha=alpha;
  (*a).p = p;
  (*a).q = q;
  (*a).np = np;
  (*a).nq = nq;
  (*a).s = 0;   /* CHANGED FROM 1 to 0 LOGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG */
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
  for (j=0;j<n*i;j++) (*a).jx[j]=-exp(1000); /* LOGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG */
  if (y!=NULL) {
	  (*a).jy=(double*) calloc(n*i,sizeof(double));
      for (j=0;j<n*i;j++) (*a).jy[j]=-exp(1000); /* LOGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG */
  }
  else (*a).jy=NULL;
  (*a).D=(int*) calloc(i,sizeof(int));
  (*a).U=(int*) calloc(MAX*i,sizeof(int));
  (*a).heap = MAX+1;


  (*a).xn=(double*) malloc(sizeof(double)*n*(MAX+1));
  for (i=0; i<n; i++) {
    (*a).jx[i]=0;              /* LOGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG */
    (*a).xn[(MAX+1)*i]=0;
    for (j=1;j<MAX+1;j++) (*a).xn[(MAX+1)*i+j]=(*a).xn[(MAX+1)*i+j-1]+log(x[i]);
  }

  if (y!=NULL) {
     (*a).yn=(double*) malloc(sizeof(double)*n*(MAX+1));
     for (i=0; i<n; i++) {
       (*a).jy[i]=0;
       (*a).yn[(MAX+1)*i]=0;
       for (j=1;j<MAX+1;j++) (*a).yn[(MAX+1)*i+j]=(*a).yn[(MAX+1)*i+j-1]+log(y[i]);
 	 }
  }

  (*a).z  =(double*)malloc((MAX+1) * sizeof(double));
  (*a).z[1]=1;
  (*a).l  =(int*)calloc(MAX+1,sizeof(int));
  (*a).lc =(int*)calloc(MAX+1,sizeof(int));
  (*a).m  =(int*)calloc(MAX+1,sizeof(int));
  (*a).mc =(int*)calloc(MAX+1,sizeof(int));

  loghg_summation(1,MAX,a);

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

  *s=(*a).s;
  free(a);
}
