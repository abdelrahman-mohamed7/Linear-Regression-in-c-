#include <iostream>
#include <immintrin.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
using namespace std;

#define MAX_ITERATION 1000 
#define Lr 0.01

typedef struct{
    vector<float> X; // feature
    vector<float> y; // target
    vector<string> meta; // metadata
    int size_;
}Data;




class numcpp {

    /*
        This class is for Vectorizing operations on arrays.
        It uses intrinsics which are bulit-in functions implemented as compiler-specific extensions 
        based on SIMD(single instruction multiple Data) which is used for parallel processing.  
    
    
    */
public:
    /*
    this method applys element-wise mul.

    --> for using if-statement : 
        1 - __m256 can only load vector of 8 single precision which are (32-bit) "floats" at time
        2 - when the size of data != any multiple(8) the rest chunks of data must be handled separately as normal instructions. 
    */
    float* mul(float* x, float* y, int n) {
        float result = 0.0f;
        int SizeRest_ = n;
        float* c = new float[n];
        for (int i = 0; i < n; i += 8) {
            SizeRest_ -= 8;
            if (SizeRest_ < 8 && SizeRest_ != 0) {
                int rest = n % 8;
                int index = n - rest;

                for (int inner = index; inner < n; inner++) {
                    c[inner] = x[inner] * y[inner];
                }

          
            }
            else {
                
                __m256 v1 = _mm256_loadu_ps(&x[i]);
                __m256 v2    = _mm256_loadu_ps(&y[i]);
                __m256 mul_vec = _mm256_mul_ps(v1, v2);

                _mm256_storeu_ps(&c[i], mul_vec);
                
            }
        }
        return c;
    }

    float dot(float *x ,float *y , int n ) {
        /*
            this method does dot product.
        */
        float result = 0.0f;
        int SizeRest_ = n;
        for (int i = 0; i < n; i += 8) {
            SizeRest_ -= 8;
            if (SizeRest_ < 8 && SizeRest_ != 0) {
                int rest = n % 8;
                int index = n - rest;
                vector<int> therest;
                for (int inner = index; inner < n; inner++) {
                    therest.push_back(x[index] * y[index]);

                }
                for (size_t k = 0; k < therest.size(); k++)
                {
                    result += therest.at(k);

                }
            }
            else {
                
                __m256 v1 = _mm256_loadu_ps(&x[i]);
                __m256 v2 = _mm256_loadu_ps(&y[i]);
                
                __m256 mul_vec = _mm256_mul_ps(v1, v2);
                __m256 sum_vec = _mm256_hadd_ps(mul_vec, mul_vec);
                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);

                __m128 sum = _mm256_extractf128_ps(sum_vec, 0);
                sum = _mm_add_ps(sum, _mm256_extractf128_ps(sum_vec, 1));
                result += _mm_cvtss_f32(sum);
            }
        }
        return result;
    }

    float* add(float* x, float* y, int n) {
        /*
           this method  adds  two vectors.
       */
        float* c = new float[n];
        int SizeRest_ = n;
        for (int i = 0; i < n; i += 8) {
            SizeRest_ = SizeRest_ - 8;
            if (SizeRest_ < 8 && SizeRest_ != 0) {
                int rest = n % 8;
                int index = n - rest;
                for (int inner = index; inner < n; inner++) {
                    c[inner] = x[inner] + y[inner];

                }

            }
            else {
               
                __m256 v1 = _mm256_loadu_ps(&x[i]);
                __m256 v2 = _mm256_loadu_ps(&y[i]);

                __m256 c_vec = _mm256_add_ps(v1, v2);

                _mm256_storeu_ps(&c[i], c_vec);
            }
        }
       
        return c;

    }

    float* sub(float *x , float *y , int n) {

        /*
          this method  substracts  two vectors.
      */
        float* c = new float[n];
        int SizeRest_ = n;
        for (int i = 0; i < n; i += 8) {

            SizeRest_ = SizeRest_ - 8;
            if (SizeRest_ < 8 && SizeRest_ != 0) {
                int rest = n % 8;
                int index = n - rest;
                for (int inner = index; inner < n; inner++) {
                    c[inner] = x[inner] - y[inner];

                }

            }
            else {
                
                __m256 v1 = _mm256_loadu_ps(&x[i]);
                __m256 v2 = _mm256_loadu_ps(&y[i]);

                __m256 c_vec = _mm256_sub_ps(v1, v2);

                _mm256_storeu_ps(&c[i], c_vec);
            }
        }

        return c;

    }


    float* pow_(float *x ,int n , int power = 2) {

        /*
            It does power operation for every element in the array.
        */

        float* c = new float[n];
        __m256 p_vec = _mm256_set1_ps(power);
        int SizeRest_ = n;
        for (int i = 0; i < n; i += 8) {
            __m256 v1 = _mm256_loadu_ps(&x[i]);
            SizeRest_ -=  8;
            if (SizeRest_ < 8 && SizeRest_ != 0) {
                int rest = n % 8;
                int index = n - rest;
                for (int inner = index; inner < n; inner++) {
                    c[inner] = pow(x[inner], 2);

                }
            }
            else {
                __m256 result_vec = _mm256_pow_ps(v1, p_vec);
                _mm256_storeu_ps(&c[i], result_vec);
            }
        }
        return c;
    }

    float* broadcasting_mul(float* x,float num, int n) {
        /*
        
            it broadcasting mul operation with an element which is num;
        */
        float* c = new float[n];
        __m256 v2 = _mm256_set1_ps(num);
        int SizeRest_ = n;
        for (int i = 0; i < n; i += 8) {
           
            __m256 v1 = _mm256_loadu_ps(&x[i]);
            SizeRest_ -= 8;
            if (SizeRest_ < 8 && SizeRest_ != 0) {
                int rest = n % 8;
                int index = n - rest;
                for (int inner = index; inner < n; inner++) {
                    c[inner] = x[inner] * num;

                }
            }
            else {
               
                __m256 result_vec = _mm256_mul_ps(v1, v2);
                _mm256_storeu_ps(&c[i], result_vec);
            }
        }
        return c;
    }
    
    float sum(float* x, int n) {
        /*
            get sum of the array 
        */

        float c =0;
        int SizeRest_ = n;
        for (int i = 0; i < n; i += 8) {
        
            __m256 v1 = _mm256_loadu_ps(&x[i]);
            SizeRest_ -= 8;
            if (SizeRest_ < 8 && SizeRest_ != 0) {
                int rest = n % 8;
                int index = n - rest;
                for (int inner = index; inner < 12; inner++) {
                    c += x[inner];

                }
            }
            else {

                /*
                    it does horizontal add
                */
                __m256 sum_vec = _mm256_hadd_ps(v1, v1);
                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);

                __m128 sum = _mm256_extractf128_ps(sum_vec, 0);
                sum = _mm_add_ps(sum, _mm256_extractf128_ps(sum_vec, 1));
                c += _mm_cvtss_f32(sum);
            }
        }
        return c;
    }

};


class LinearReg {

private:
    Data data;
    int n;
    float*x ;
    float* y ;
    float* weights;
    float* bias;
    numcpp nc;

public:
    float* loss;
    LinearReg(Data data) {
        this->data = data;
        this->n = this->data.size_;
        this->x = new float[n];
        this->y = new float[n];
        this->loss = new float[MAX_ITERATION];
        this-> weights = new float[n];
        this->bias = new float[n];
        this->x = this->data.X.data();
        this->y = this->data.y.data();  
        for (size_t i = 0; i < n; i++)
        {
            this->weights[i] = 0;
            this->bias[i] = 0;

        }
    } 
  


    void fit() {
        /*
            Update weights and bias 
        
        */
        for (size_t i = 0; i < MAX_ITERATION; i++)
        {
            this->update();
            loss[i] = this->losss();
        }
        cout << "The weight  : "<<weights[0] << endl;
        cout <<"The bias : " << bias[0] << endl;


    }

    float losss() {
        float* yhat = this->nc.sub(this->y, this->predict(), this->n);
        float* pow = this->nc.pow_(yhat, this->n);
        float sums = this->nc.sum(pow, this->n);

        return 1.0/(2*this->n) * sums ;
    }

    void update() {
        float* yhat = this->nc.sub(this->y ,this->predict(), this->n);
        float dw = -(2 * (this->nc.dot(this->x, yhat, this->n))) / this->n;
        float db = -2*this->nc.sum(yhat, this->n)/this->n;

       
        for (size_t i = 0; i < n; i++)
        {
            this->weights[i] -= Lr*dw ;
            this->bias[i] -= Lr*db;

        }

        
        return ;

    }
 




    float*  predict() {
       float* XW = new float[this->n];
       float* y_hat = new float[this->n];
       XW = this->nc.mul(this->x, this->weights, this->n);
       y_hat = this->nc.add( XW,this->bias ,this->n);
       return y_hat;
    }

         



};





class csv
{

    private:
        char delimiter = ',';
        Data data;
        string filepath;


    public:

        csv(string filepath) {
            this->filepath = filepath;
        }
       

        Data Get_data() {
            int metaflag_ = 0;
            string mainLine_, values;
            ifstream csv(this->filepath);
            while (getline(csv, mainLine_)) {
                stringstream cop_(mainLine_);
                string value;
                if (metaflag_ == 0) {
                    metaflag_ = 1;
                    while (getline(cop_, values, delimiter)) {
                        this->data.meta.push_back(values);}
                }
                int valueflag_ = 0;
                while (getline(cop_, values, delimiter)) {
                    if (valueflag_ == 0) {
                        this->data.X.push_back(stof(values));
                        valueflag_ = 1;
                    }
                    else
                    {
                        this->data.y.push_back(stof(values));
                    }
                    
                }

                

            }

            this->data.size_ = this -> data.X.size();
            csv.close();

            return this->data;
        }




};





int main()
{
    csv read("data.csv");

    Data get = read.Get_data();

    float*x = get.X.data();
    float* y = get.y.data();
    int s = get.size_;

    LinearReg linear(get);
    linear.fit();
    
    float* ss = linear.loss;
    cout << "The starting loss : " << ss[0] << endl;
    cout << "The ending loss : " << ss[MAX_ITERATION-1];

return 0;
}