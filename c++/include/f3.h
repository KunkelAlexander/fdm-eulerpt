double dot(double a, double b) { return a * b; };
double k1, k2, k3, k4, eta0, eta;
double pow(double a, int b) { return a * b; };


double s12 = dot(k1 + k2, k1 + k2);
double s23 = dot(k2 + k3, k2 + k3);
double s13 = dot(k1 + k3, k1 + k3);

double s11 = dot(k1, k1);
double s22 = dot(k2, k2);
double s33 = dot(k3, k3);

double b1, b2, b3;

if (s23 < verysmalleps) 
{
    b1 = 0;
}
else 
{
b1 = (2 * dot(k2, k3) * dot(k2 + k3, k2 + k3) * ((35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k2 + k3, k2 + k3) * dot(k1 + k2 + k3, k1) + 10 * (7 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 2 * pow(eta0, 9)) * dot(k1, k1) * dot(k1 + k2 + k3, k2 + k3) + 4 * (5 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 9 * pow(eta, 2) * pow(eta0, 7) - 5 * pow(eta0, 9)) * dot(k1, k2 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k3, k3) * dot(k2 + k3, k2) * ((175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k2 + k3, k2 + k3) * dot(k1 + k2 + k3, k1) + 3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k1, k1) * dot(k1 + k2 + k3, k2 + k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k1, k2 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k2, k2) * dot(k2 + k3, k3) * ((175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k2 + k3, k2 + k3) * dot(k1 + k2 + k3, k1) + 3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k1, k1) * dot(k1 + k2 + k3, k2 + k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k1, k2 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3))) / dot(k2 + k3, k2 + k3);
}

if (s13 < verysmalleps) 
{
    b2 = 0;
}
else
{
b2 = (2 * dot(k1, k3) * dot(k1 + k3, k1 + k3) * ((35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k1 + k3, k1 + k3) * dot(k1 + k2 + k3, k2) + 10 * (7 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 2 * pow(eta0, 9)) * dot(k2, k2) * dot(k1 + k2 + k3, k1 + k3) + 4 * (5 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 9 * pow(eta, 2) * pow(eta0, 7) - 5 * pow(eta0, 9)) * dot(k2, k1 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k3, k3) * dot(k1 + k3, k1) * ((175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k1 + k3, k1 + k3) * dot(k1 + k2 + k3, k2) + 3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k2, k2) * dot(k1 + k2 + k3, k1 + k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k2, k1 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k1, k1) * dot(k1 + k3, k3) * ((175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k1 + k3, k1 + k3) * dot(k1 + k2 + k3, k2) + 3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k2, k2) * dot(k1 + k2 + k3, k1 + k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k2, k1 + k3) * dot(k1 + k2 + k3, k1 + k2 + k3))) / dot(k1 + k3, k1 + k3);
}

if (s12 < verysmalleps) 
{
    b3 = 0;
}
else 
{
b3 = (2 * dot(k2, k1) * dot(k1 + k2, k1 + k2) * (10 * (7 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 2 * pow(eta0, 9)) * dot(k3, k3) * dot(k1 + k2 + k3, k1 + k2) + (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k1 + k2, k1 + k2) * dot(k1 + k2 + k3, k3) + 4 * (5 * pow(eta, 9) - 9 * pow(eta, 7) * pow(eta0, 2) + 9 * pow(eta, 2) * pow(eta0, 7) - 5 * pow(eta0, 9)) * dot(k3, k1 + k2) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k2, k2) * dot(k1 + k2, k1) * (3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k3, k3) * dot(k1 + k2 + k3, k1 + k2) + (175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k1 + k2, k1 + k2) * dot(k1 + k2 + k3, k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k3, k1 + k2) * dot(k1 + k2 + k3, k1 + k2 + k3)) + dot(k1, k1) * dot(k1 + k2, k2) * (3 * (35 * pow(eta, 9) - 90 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) - 8 * pow(eta0, 9)) * dot(k3, k3) * dot(k1 + k2 + k3, k1 + k2) + (175 * pow(eta, 9) - 270 * pow(eta, 7) * pow(eta0, 2) + 63 * pow(eta, 5) * pow(eta0, 4) + 32 * pow(eta0, 9)) * dot(k1 + k2, k1 + k2) * dot(k1 + k2 + k3, k3) + 6 * (5 * pow(eta, 9) - 18 * pow(eta, 7) * pow(eta0, 2) + 21 * pow(eta, 5) * pow(eta0, 4) - 12 * pow(eta, 2) * pow(eta0, 7) + 4 * pow(eta0, 9)) * dot(k3, k1 + k2) * dot(k1 + k2 + k3, k1 + k2 + k3))) / dot(k1 + k2, k1 + k2)
}
double x = (b1 + b2 + b3) / (3780. * pow(eta, 9) * dot(k1, k1) * dot(k2, k2) * dot(k3, k3));