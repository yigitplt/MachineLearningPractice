#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class LinearRegression {
private:
    vector<vector <double> > X;
    vector<double> Y;
    vector<double> w;
    double b;
    double lambda;


    /**
     * Struct to hold gradients of weight and bias for optimization
     */
    struct Gradient{
        vector<double> dj_dw;
        double dj_db;
    };


    /**
     * Computes the dot product of two vectors.
     * 
     * @param A First vector.
     * @param B Second vector.
     * @return The dot product of A and B.
     */
    double dot(const vector<double> A, const vector<double> B) const {

        double result = 0;
        
        for(int i = 0; i < A.size(); i++){
            result += A[i] * B[i];
        }

        return result;
    }


    /**
     * Computes the cost function with optional regularization.
     * 
     * @return The computed cost value.
     */
    double computeCost() const {
        int m = X.size();
        double cost = 0;
        double regCost = 0;
        double f = 0;

        for(int i = 0; i < m; i++){
            f = dot(w, X[i]) + b;
            cost += (f - Y[i]) * (f - Y[i]);
        }
        cost /= (2*m);

        for(int i = 0; i < w.size(); i++){
            regCost += w[i] * w[i];
        }
        regCost *= lambda/(2*m);

        return cost + regCost;

    }


    /**
     * Computes the gradient of the cost function with respect to weights and bias.
     * 
     * @return A Gradient struct containing gradients of weights and bias.
     */
    Gradient computeGradient() {

        int m = X.size();
        int n = X[0].size();
        double f = 0.0;
        vector<double> dj_dw(n, 0);
        double dj_db = 0;

        for(int i = 0; i < m; i++){
            f = dot(w, X[i]) + b;
            for(int j = 0; j < n; j++){
                dj_dw[j] += (f - Y[i]) * X[i][j];
            }
            dj_db += f - Y[i];
        }

        for(int j = 0; j < n; j++){
            dj_dw[j] = (dj_dw[j] / m) + (lambda/m) * w[j];
        }

        dj_db /= m;

        return {dj_dw, dj_db};

    }


    /**
     * Prints the final model parameters (weights and bias).
     * 
     * @param w The weights vector.
     * @param b The bias term.
     */
    void printParameters(vector<double> w, double b){
        cout << "Final w: " << "[";
        for (size_t i = 0; i < w.size(); i++) {
            cout << w[i];
            if (i != w.size() - 1) cout << ", "; // Add commas between elements
        }
        cout << "]" << endl;
        cout << "Final b: " << b << endl;

            
    }


public:
    LinearRegression(const vector<vector<double> >& x, const vector<double>& y, const double reg_lambda = 0.0) : 
    X(x), Y(y), w(X[0].size(), 0.0), b(0), lambda(reg_lambda) {}


    /**
     * Trains the linear regression model using gradient descent.
     * 
     * @param alpha Learning rate for gradient descent.
     * @param numIters Number of iterations for training.
     */
    void fit(double alpha, int numIters) {

        vector<double> weight = w;
        double bias = b;

        for(int i = 0; i <= numIters; i++){
            Gradient gradient = computeGradient();

            for(int j = 0; j < w.size(); j++){
                w[j] = w[j] - alpha * gradient.dj_dw[j];
            }

            b = b - alpha * gradient.dj_db;

            double cost = computeCost();

            if(i % 500 == 0){
                cout << "Iteration: " << i << " Cost: " << cost << endl;
            }
            
        }

        printParameters(w, b);
    }


    /**
     * Makes a prediction using the trained model.
     * 
     * @param x A single feature vector for prediction.
     * @return The predicted value.
     */
    double predict(vector<double> x) const {
        return dot(x , w) + b;
    }
};


int main() {
    vector<vector<double> > X = {{1, 2}, {2, 3}, {3, 2}, {4, 5}, {5, 6}, {8,4}};
    vector<double> Y = {12, 17, 18, 27, 32, 37}; // 3X_1 + 2X_2 + 5

    double alpha = 0.01;
    double lambda = 0;
    double iterationCount = 5000;

    LinearRegression lr(X, Y);
    lr.fit(alpha , iterationCount);
    
    vector<double> x = {5, 2};
    cout << "Prediction for the given data" << ": " << lr.predict(x) << endl;

    return 0;
}