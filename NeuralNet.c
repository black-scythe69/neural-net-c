#include<stdio.h>
#include<math.h>
#include<stdlib.h>

struct matrix {
int rows;
int cols;
float *data;
};
struct matrix *x, *y, *x1, *x2, *w1, *w1T, *w2, *w2T, *b1, *b2, 
*a1, *a1T, *a2, *a2T, *z1, *z2, *del1, *del2, *gradW1, *gradW2, 
*gradB1, *gradB2, *E, *yT, *xT;
float n=1.5;

struct matrix *createMatrix(int rows, int cols){
struct matrix *m=malloc(sizeof(struct matrix));
if(m==NULL){
printf("Memory Allocation Failed\n");
exit(0);
}
m->rows=rows;
m->cols=cols;
m->data=calloc(rows*cols, sizeof(float));
return m;
}

void fill_weights(struct matrix* W){
int i;
for(i=0; i<(W->cols*W->rows); i++){
W->data[i]=((float)rand()/RAND_MAX) * 2.0 - 1.0;
}
}

void fill_bias(struct matrix* B){
int i;
for(i=0; i<(B->cols*B->rows); i++){
B->data[i]=((float)rand()/RAND_MAX) * 2.0 - 1.0;
}
}

struct matrix *multiply(struct matrix *a, struct matrix *b){
struct matrix *R=createMatrix(a->rows, b->cols);
int i, j, k;
for(i=0; i<(a->rows); i++){
for(j=0; j<(b->cols); j++){
for(k=0; k<(b->rows); k++){
R->data[i*R->cols+j]+= a->data[i*a->cols+k] 
* b->data[k*b->cols+j];
}
}
}
return R;
}

struct matrix *scalar(struct matrix *A, float n){
struct matrix *s=createMatrix(A->rows, A->cols);
int i, j;
for(i=0; i<A->rows; i++){
for(j=0; j<A->cols; j++){
s->data[i*A->cols+j]= n*A->data[i*A->cols+j];
}
}
return s;
}

struct matrix *add(struct matrix *a, struct matrix *b){
struct matrix *R= createMatrix(a->rows, b->cols);
int i;
for(i=0; i<(a->rows*a->cols); i++)
 R->data[i]=a->data[i]+b->data[i];
return R;
}

struct matrix *subtract(struct matrix *a, struct matrix *b){
struct matrix *R=createMatrix(a->rows, a->cols);
int i;
for(i=0; i<(a->rows*a->cols); i++)
 R->data[i]=a->data[i]-b->data[i];
return R;
}

void activate(struct matrix *zh, struct matrix* ah){
int i;
for(i=0; i<(zh->rows*zh->cols); i++)
 ah->data[i] = 1.0 / (1.0+exp(-zh->data[i]));
}

void Loss( struct matrix *yo, struct matrix *yt, struct matrix 
*E){
int i;
for(i=0; i<(yo->rows*yo->cols); i++){
E->data[i]=yo->data[i]- yt->data[i];
}
}

void transpose(struct matrix *T, struct matrix *A){
 int i, j;
 for(i=0; i<A->rows; i++){
 for(j=0; j<A->cols; j++){
 T->data[j*A->rows+i]=A->data[i*A->cols+j];
}
}
}

void assign(struct matrix *A, struct matrix *B){
int i;
for(i=0; i<(A->rows*A->cols); i++){
A->data[i]=B->data[i];
}
}

void forward(){
z1=add(multiply(w1, x), b1);
activate(z1, a1);
assign(x2, a1);
z2=add(multiply(w2, x2),b2);
activate(z2, a2);
Loss(a2, yT, E);
}

void backProp(){
transpose(a1T, a1);
transpose(xT, x);
transpose(w2T, w2);
assign(del2, subtract(a2, yT));
for(int i=0; i<del2->rows; i++){
   del2->data[i] *= a2->data[i] * (1.0 -     a2->data[i]);
}
assign(gradW2, multiply(del2, a1T));
assign(gradB2, del2);
assign(del1, multiply(w2T, del2));
for(int i=0; i<del1->rows; i++){
  del1->data[i] *= a1->data[i] * (1.0 - a1->data[i]);
}
assign(gradW1, multiply(del1, xT));
assign(gradB1, del1);
}

void update(){
assign(w2, subtract(w2, scalar(gradW2, n)));
assign(b2, subtract(b2, scalar(gradB2, n)));
assign(w1, subtract(w1, scalar(gradW1, n)));
assign(b1, subtract(b1, scalar(gradB1, n)));
}

void main(){
printf("Program started\n");
x= createMatrix(2, 1);
y= createMatrix(1, 1);
w1= createMatrix(2, 2);
w1T= createMatrix(2, 2);
w2= createMatrix(1, 2);
w2T= createMatrix(2, 1);
b1= createMatrix(2, 1);
b2= createMatrix(1, 1);
a1= createMatrix(2, 1);
a1T= createMatrix(1, 2);
a2= createMatrix(1, 1); 
a2T= createMatrix(1, 2);
z1= createMatrix(2, 1);
z2= createMatrix(1, 1);
del1= createMatrix(2, 1);
del2= createMatrix(1, 1);
gradW1= createMatrix(2, 2);
gradW2= createMatrix(1, 2);
gradB1= createMatrix(2, 1);
gradB2= createMatrix(1, 1);
E= createMatrix(1, 1);
yT= createMatrix(1, 1);
xT= createMatrix(1, 2);
x1=createMatrix(2, 1);
x2=createMatrix(2, 1);

fill_weights(w1);
fill_weights(w2);
fill_bias(b1);
fill_bias(b2);

printf("Matrices initialised\n");
float xor_x[4][2]={{0, 0},{0, 1},{1, 0},{1, 1}};
float xor_y[4]={0, 1, 1, 0};
int epochs = 15000;
int e, s;

for(e=0; e<=epochs; e++){
float total_loss=0;
for(s=0; s<4; s++){
x->data[0]= xor_x[s][0];
x->data[1]= xor_x[s][1];
yT->data[0]= xor_y[s];
forward();
backProp();
update();
total_loss+=(E->data[0]*E->data[0]);
}
if(e%250==0)
 printf("Epoch %d, avg loss: %f\n ", e, 
total_loss/4);
if(total_loss/4 < 0.001)
 break;
}

float test_x[4][2] = {{0,0},{0,1},{1,0},{1,1}};
 int t;
 for(t=0; t<4; t++){
 x->data[0] = test_x[t][0];
 x->data[1] = test_x[t][1];
 forward();
 printf("[%d,%d] -> %f\n", (int)test_x[t][0], 
(int)test_x[t][1], a2->data[0]);
 }
}