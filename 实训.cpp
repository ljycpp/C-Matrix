#include <iostream>
#include <vector>       //C++ ��׼���е� vector �����Ķ������ع������뵽���ĳ����С��洢��������ݣ� vector<vector<T>> data_
#include <fstream>      //���ļ����ж�д��������ͺ�����
#include <complex>      //���Խ��и��������������
#include <stdexcept>    //�����׼�쳣���
#include <cmath>        //������ѧ�����ͳ���������sqrt������ֵabs
#include <algorithm>    //�����㷨��,swap��max����
#include <sstream>      // ������ͷ�ļ�֧���ַ�������ͨ�� istringstream ��һ���ַ����ָ�ɶ�����ݣ����������ȡ�ʹ������Ԫ�ء�
#include <limits>       // ������ͷ�ļ�֧��numeric_limits�����ڼ�������������С������
#include <string>       // ȷ������stringͷ�ļ�
using namespace std;    //ʹ�ñ�׼�����ռ�
const static constexpr double EPSILON = 1e-12;//���һ����С�����֣������ж��������Ƿ�Ϊ��
// ���ࣺ������󹫹��ӿڣ���̬��
class MatrixBase
{
public:
    virtual ~MatrixBase() = 0;// ������������
    virtual void print() const = 0;//��ӡ���󣬴��麯��
    virtual void saveToFile(const string& filename) const = 0;// // ���浽�ļ������麯��
    virtual size_t getRows() const = 0;//��ȡ����
    virtual size_t getCols() const = 0;//��ȡ����
};
// �������������Ķ���
MatrixBase::~MatrixBase() {}
// ģ�������
template<typename T>
class Matrix : public MatrixBase
{
private:
    size_t rows_;//���ڵ�������������� size_t ��һ���޷����������ͣ����ڱ�ʾ�Ǹ�����
    size_t cols_;//��
    vector<vector<T>> data_;//��� vector ���У��ڲ� vector ���С�

public:
    //��������������أ���Ԫ����������
    template<typename T>//��������Ԫ����֧�ֲ�ͬ����ģ�����������
    friend istream& operator>>(istream& ifs, Matrix<T>& mat);
    template<typename T>
    friend ostream& operator<<(ostream& ofs, Matrix<T>& mat);
    // ���캯��
    Matrix() : rows_(0), cols_(0), data_() {}  //�޲�Ĭ�Ϲ��캯������Ϊ0��0��
    Matrix(size_t rows, size_t cols, const T& init = T())//ָ��Ԫ�صĳ�ʼֵ��Ĭ��ΪT���͵�Ĭ��ֵ�����ô��ݿ��Ա��ⴴ������ĸ���
        : rows_(rows), cols_(cols), data_(rows, vector<T>(cols, init)) {
    }
    //ָ���������ͳ�ʼֵ�Ĺ��캯��������һ��rows�еĶ�̬����ÿ��Ԫ����cols�У���ʼֵΪinit
    Matrix(const vector<vector<T>>& data)//�������ñ��ⲻ��Ҫ�Ŀ���
        : rows_(data.size()), cols_(data.empty() ? 0 : data[0].size()), data_(data) {
    }
    // ���������������Ϊ����Ķ�ά����������С��������Ϊ��һ�е�Ԫ�ظ��������������еĳ�����ͬ����ֱ�ӽ�����Ķ�ά������ֵ��������ڲ����ݳ�Ա data_
 //vector ֧�ֿ�����ֵ �����Կ���ֱ��data_(data)
 // �������캯��
    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}
    // �����麯��ʵ��
    void print() const override//�������Ǵ��麯������д��const�Ǻ���ǩ����һ���֡�
    {    //auto��ʾ�Զ������Ƶ������з�Χ����ѭ��
        for (const auto& row : data_) //����data_�������Ԫ��
        {
            for (const T& elem : row)//�������������Ԫ��
            {
                if constexpr (is_same_v<T, complex<double>>)//���������﷨�������һ���������������к����
                {
                    double real = elem.real();//����real() - ���ظ�����ʵ��
                    double imag = elem.imag();//����imag() - ���ظ�������
                    // ����ȫ�������ʵ���鲿ȫΪ0�����
                    if (abs(real) <= EPSILON && abs(imag) <= EPSILON)
                    {
                        cout << "0";
                    }
                    // ����ʵ�����鲿Ϊ0�����
                    else if (abs(real) > EPSILON && abs(imag) <= EPSILON)
                    {
                        cout << real;
                    }
                    // ����������ʵ��Ϊ0�����
                    else if (abs(real) <= EPSILON && abs(imag) > EPSILON)
                    {
                        if (abs(abs(imag) - 1.0) <= EPSILON) //�鲿����ֵΪ1��ʡ��+-1
                        {
                            cout << (imag > 0 ? "" : "-") << "i";
                        }
                        else
                        {
                            cout << imag << "i";
                        }
                    }
                    // �������ʵ�������鲿�ĸ���
                    else
                    {
                        cout << real;
                        // ���Ŵ����Ż�
                        cout << (imag > 0 ? "+" : "-");
                        double abs_imag = abs(imag);
                        if (abs(abs_imag - 1.0) <= EPSILON)
                        {
                            cout << "i";  // ���鲿����ֵΪ1ʱ��ʡ��+-1
                        }
                        else
                        {
                            cout << abs_imag << "i";
                        }
                    }
                }
                // �������ͱ���ԭ���߼�
                else if constexpr (is_same_v<T, double> || is_same_v<T, float>)
                {
                    if (abs(elem) < EPSILON)
                    {
                        cout << "0";
                    }
                    else
                    {
                        cout << elem;
                    }
                }
                else//ʣ��������������ֱ�����
                {
                    cout << elem;
                }
                cout << "\t";//�Ʊ�����ڶ������
            }
            cout << endl;
        }
    }
    size_t getRows() const override
    {
        return rows_;
    }//��д�麯�����õ�����
    size_t getCols() const override
    {
        return cols_;
    }//��д�麯�����õ�����

    // Part 1�����ظ������������
     // ���ظ�ֵ�����
    Matrix& operator=(const Matrix& other)
    {
        if (this != &other)
        {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = other.data_;
        }
        return *this;
    }
    //����ӷ����㷨�����жϾ��������Ƿ���ͬ��Ȼ����ж�Ӧλ��[i][j]�����
    Matrix operator+(const Matrix& other) const
    {
        checkSameDimensions(other);//���ù��ߺ��������ά���Ƿ���ͬ���������ͬ���׳��쳣
        Matrix result(rows_, cols_);//�´���һ������ÿ��λ�õ���ԭ�����������Ԫ�����
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result.data_[i][j] = data_[i][j] + other.data_[i][j];
            }
        }
        return result;
    }
    //����������㷨�����жϾ��������Ƿ���ͬ��Ȼ����ж�Ӧλ��[i][j]�������ʵ�ֹ������Ƽӷ�
    Matrix operator-(const Matrix& other) const
    {
        checkSameDimensions(other);
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result.data_[i][j] = data_[i][j] - other.data_[i][j];
            }
        }
        return result;
    }
    //����˷�����ָ����������ˣ��㷨��
    // ���ж�A�����Ƿ����B���У�Ȼ��result��[i][j]����A��i�ж�ӦԪ�س�B��j�еĶ�ӦԪ�صĻ����
    Matrix operator*(const Matrix& other) const
    {
        if (cols_ != other.rows_)
        {
            throw invalid_argument("ά�Ȳ�ƥ�䣬�޷����");//�����ж�
        }
        Matrix result(rows_, other.cols_);//���ù��캯������ʼ��Ϊ��
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t k = 0; k < cols_; ++k) //A.cols_=B,rows_
            {
                for (size_t j = 0; j < other.cols_; ++j)
                {
                    result.data_[i][j] += data_[i][k] * other.data_[k][j];
                }
            }
        }
        return result;
    }
    // ��Ӿ����������˵������,�㷨����Ӧλ�õ�Ԫ�س��Ա���
    Matrix operator*(const T& scalar) const
    {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result.data_[i][j] = data_[i][j] * scalar;
            }
        }
        return result;
    }
    // �����Ԫ������֧�ֱ����˾���Ľ����ɣ�2*A=A*2
    // ����Ϊ��Ԫ��Ҫ��Ϊ���ܹ�ʹ����������ͬ������������﷨
    friend Matrix operator*(const T& scalar, const Matrix& mat)
    {
        return mat * scalar; // �������еľ���˱��������
    }
    //����==��������ж����������Ƿ���ȫ��ͬ���㷨���Ծ����Ӧλ�õ�ÿһ��Ԫ�ؽ��бȽ�
    bool operator==(const Matrix& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)//�Ƚ��������ж�
        {
            return false;
        }
        for (size_t i = 0; i < rows_; ++i) //�ٽ��ж�Ӧλ�õĶ�ӦԪ�صĴ�С�ж�
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                if (abs(data_[i][j] - other.data_[i][j]) > EPSILON) return false;
            }
        }
        return true;
    }
    // Part 2����������
    // ����ת��
    // ���壺����������к��л���������һ�� m��n �ľ��� A����ת�þ��� A^T ��һ�� n��m �ľ���
    /* �㷨��1.����һ���¾�������Ϊԭ���������������Ϊԭ���������
             2. ����ԭ�����ÿ��Ԫ��
             3. ��ԭ������λ��(i, j) ��Ԫ�طŵ��¾����λ��(j, i)*/
    Matrix<T> transpose() const
    {
        Matrix result(cols_, rows_);//�����¾���
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result.data_[j][i] = data_[i][j];//����λ��
            }
        }
        return result;
    }
    // ��ӹ���ת�÷���
    Matrix<T> conjugateTranspose() const
    {
        Matrix<T> result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                if constexpr (is_same_v<T, complex<double>>)
                    //����Ǹ������ͣ���ȡ����
                {
                    result.data_[j][i] = conj(data_[i][j]);// ʹ��conj������ȡ�����Ĺ��
                    //����complexͷ�ļ��ṩ�ĺ���
                }
                else
                {
                    result.data_[j][i] = data_[i][j];
                }
            }
        }
        return result;
    }
    //��������
    //��ѧ���壺��������ʽ�Ľ��� �����������ķ�������ʽ�Ľ���
    //�㷨�����ø�˹��Ԫ�����Ѿ���ת��Ϊ������ʽ�����շ����е��������Ǿ�����ȡ�
    //���д��� �������ұ���ÿһ��
    //Ѱ����Ԫ ���ڵ�ǰ�����ҵ���һ������Ԫ����Ϊ��Ԫ
    //�н��� ������Ԫ�������뵱ǰ������н���
    //�й�һ�� ������Ԫ�����е���Ԫ��һ��Ϊ1
    //��Ԫ ����ȥ��Ԫ���·�������Ԫ��
    //���� ��ÿ�ҵ�һ����Ԫ���ȼ�1����ͬʱ��ʾ��һ�ִ���һ�п�ʼ������������Ȼ�����
    int rank() const
    {
        Matrix mat = *this;// �������󸱱������ı�ԭ����
        int rank = 0;
        for (size_t col = 0; col < cols_ && rank < rows_; ++col) //��С�ڵ�������
        {
            size_t pivot = findPivot(mat, col, rank);
            if (pivot == rows_)
            {
                continue;
            }
            swapRows(mat, rank, pivot);
            normalizeRow(mat, rank, col);
            eliminateRows(mat, rank, col);
            rank++;
        }
        return rank;
    }
    //������ת��Ϊ�������ʽ
    /*������ε��ص�:
    ÿ�������еĵ�һ������Ԫ�أ���Ԫ��Ϊ1
    ÿ����Ԫ�����е�����Ԫ�ض�Ϊ0
    ��Ԫ�����ҡ����ϵ�������
    ����ȫ���ж��ھ���ײ�*/
    //�㷨���������������ƣ����д�����Ҫ�ѵ�ǰ�е�����Ԫ�ض���Ϊ0��
    // ���൱�����һ����Ԫ��һ�£����Ҳ��ü�������󷵻ؾ���������ʽ
    Matrix<T> reducedRowEchelonForm() const
    {
        Matrix mat = *this;
        int currentRow = 0;
        // ����ÿһ��
        for (size_t col = 0; col < cols_ && currentRow < rows_; ++col)
        {
            // Ѱ����Ԫ
            size_t pivot = findPivot(mat, col, currentRow);
            // �����ǰ��û�з���Ԫ�أ�����������һ��
            if (pivot == rows_)
            {
                continue;
            }
            swapRows(mat, currentRow, pivot);
            normalizeRow(mat, currentRow, col);
            eliminateAllRows(mat, currentRow, col);
            // �ƶ�����һ��
            currentRow++;
        }
        return mat;
    }
    // ���㷽������ʽ��ֵ=���Խ��ߵĳ˻�*��������
    //�㷨�����ж��ǲ��Ƿ���Ȼ����и�˹��Ԫ����Ҫע����ŵ��ж�
    T determinant() const
    {
        if (!isSquare())
        {
            throw invalid_argument("�Ƿ����޷���������ʽ");
        }
        Matrix mat = *this;
        T det = 1;//���ڼ�¼����ʽ��ֵ����ʼ��Ϊ1
        int swapCount = 0;//���ڼ�¼�н����Ĵ���
        for (size_t col = 0; col < cols_; ++col)
        {
            size_t pivot = findPivot(mat, col, col);
            if (pivot == rows_)
            {
                return 0;
            }
            if (pivot != col)
            {
                swapRows(mat, col, pivot);
                swapCount++;
            }
            det *= mat.data_[col][col];
            eliminateLower(mat, col);
        }
        return swapCount % 2 ? -det : det;
    }
    //������� ������� A^(-1) ���� A��A^(-1) = A^(-1)��A = I
    //���÷�����A^(-1) = (1/det(A)) �� adj(A)��det(A) ������ʽ��adj(A) �ǰ������
    //�㷨�����ж��ǲ��Ƿ���Ȼ���������ʽ���������ʽΪ�㣬��ʾ���󲻿��棬���ŷֱ���
    //������������ʽ����󷵻ذ�������������ʽ�Ľ��
    //ʵ�����ͺ͸������͵Ĵ���ʽ���в�ͬ������������ͨ����ģ���ж��Ƿ�Ϊ��
    auto inverse() const
    {
        if constexpr (is_same_v<T, complex<double>>) // ����Ǹ�������
        {
            if (!isSquare())
            {
                throw invalid_argument("�Ƿ��󣬲�����");
            }
            complex<double> det = this->determinant();//��������ʽ
            if (abs(det) < EPSILON)//������ʽΪ�㣬��ʾ���󲻿���
            {
                throw runtime_error("���󲻿���");
            }
            Matrix<complex<double>> adj = this->adjugate();//����������
            Matrix<complex<double>> result(this->getRows(), this->getCols());//�����������
            for (size_t i = 0; i < this->getRows(); ++i)
            {
                for (size_t j = 0; j < this->getCols(); ++j)
                {
                    result.at(i, j) = adj.at(i, j) / det;
                }
            }
            return result;
        }
        else
        {
            if (!isSquare())
            {
                throw invalid_argument("�Ƿ��󣬲�����");
            }
            T det = determinant();//��������ʽ
            if (isZero(det))
            {
                throw runtime_error("���󲻿���");
            }
            Matrix<double> adj = this->adjugate();//����������
            Matrix<double> result(this->getRows(), this->getCols());//�����������
            for (size_t i = 0; i < this->getRows(); ++i)
            {
                for (size_t j = 0; j < this->getCols(); ++j)
                {
                    result.at(i, j) = adj.at(i, j) / det;
                }
            }
            return result;
        }
    }
    // ��������
    //��ѧ���� ��������� adj(A) �ĵ� (j,i) ��Ԫ���Ǿ��� A �ĵ� (i,j) ��Ԫ�صĴ�������ʽ
    //���൱�ڴ�������ʽ�����ת��
    //�㷨�����ж��ǲ��Ƿ���Ȼ�����ÿ��λ�õĴ�������ʽ�����ת�õõ��������
    //��������ʽ��ȥ���� i �к͵� j �к���Ӿ��������ʽ������ (-1)^(i+j)
    Matrix<T> adjugate() const
    {
        if (!isSquare())
        {
            throw invalid_argument("�Ƿ���");
        }
        Matrix<T> cof(rows_, cols_);//������������ʽ����
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                Matrix<T> minor = subMatrix(i, j);//��ȡȥ���� i �к͵� j �к���Ӿ���
                T sign = ((i + j) % 2 == 0) ? T(1) : T(-1);//�������
                cof.at(i, j) = sign * minor.determinant();//�����Ӿ��������ʽ�����ŵ�������ֵ
            }
        }
        return cof.transpose(); // ��������Ǵ�������ʽ�����ת��
    }
    //�����ݱ��浽�ļ�����
    void saveToFile(const string& filename) const override
    {
        ofstream ofs(filename);//���ļ�����д��
        if (!ofs) 
        {
            throw runtime_error("�޷����ļ�");
        }
        ofs << rows_ << " " << cols_ << endl;//��д���к���
        for (const auto& row : data_)//����ÿһ�У�����д��ÿһ�е�Ԫ��
        {
            for (const T& elem : row)
            {
                ofs << elem << " ";
            }
            ofs << endl;
        }
    }
    // �������������ļ����ؾ���
    static Matrix<T> loadFromFile(const string& filename)
    {
        ifstream ifs(filename);//���ļ����ж�ȡ
        if (!ifs)
        {
            throw runtime_error("�޷����ļ�");
        }
        Matrix<T> mat;
        ifs >> mat;// ʹ�����ص������������ȡ��������
        return mat;
    }
    // ��Matrix���public���������������
    vector<complex<double>> eigenvalues(int max_iterations = 1000) const;//������������ֵ��ʱ���õ�

    // �����������㺯��
    // p-�������㣬p=1Ϊ�����ٷ�����p=2Ϊŷ����÷���
    static double vectorNorm(const vector<T>& v, double p)
    {
        // ������������ (p = inf)
        if (isinf(p)) //��������Ԫ�ؾ���ֵ�����ֵ
        {
            double maxVal = 0.0;
            for (const auto& val : v)
            {
                double absVal = 0.0;
                if constexpr (is_same_v<T, complex<double>>)
                {
                    absVal = abs(val);  // ����ֱ��ȡģ
                }
                else
                {
                    absVal = abs(static_cast<double>(val));  // ������ֵ����ת��
                }
                maxVal = max(maxVal, absVal);
            }
            return maxVal;
        }
        // ����p-������ÿ��Ԫ�ص�p�η���ͣ�����ڿ���
        double sum = 0.0;
        for (const auto& val : v)
        {
            double absVal = 0.0;
            if constexpr (is_same_v<T, complex<double>>)
            {
                absVal = abs(val);  // ����ȡģ
            }
            else
            {
                absVal = abs(static_cast<double>(val));  // ��ֵ����ת��
            }
            sum += pow(absVal, p);
        }
        return pow(sum, 1.0 / p);//����pow��������p�η���
    }
    // �����Frobenius��������������Ԫ�ؾ���ֵ��ƽ�����ٿ�����
    double frobeniusNorm() const
    {
        double sum = 0.0;
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                if constexpr (is_same_v<T, complex<double>>)
                {
                    sum += norm(data_[i][j]); // ������ģ��ƽ��
                }
                else
                {
                    double val = static_cast<double>(data_[i][j]);
                    sum += val * val;
                }
            }
        }
        return sqrt(sum);
    }

    // ������кͷ��������������ÿһ������Ԫ�ؾ���ֵ֮�͵����ֵ
    double rowSumNorm() const
    {
        double maxRowSum = 0.0;
        for (size_t i = 0; i < rows_; ++i)
        {
            double rowSum = 0.0;
            for (size_t j = 0; j < cols_; ++j)
            {
                if constexpr (is_same_v<T, complex<double>>)
                {
                    rowSum += abs(data_[i][j]);//���㸴����ģ
                }
                else
                {
                    rowSum += abs(static_cast<double>(data_[i][j]));
                }
            }
            maxRowSum = max(maxRowSum, rowSum);
        }
        return maxRowSum;
    }
    // ������кͷ�����ÿһ������Ԫ�ؾ���ֵ֮�͵����ֵ
    double columnSumNorm() const
    {
        double maxColSum = 0.0;
        for (size_t j = 0; j < cols_; ++j)
        {
            double colSum = 0.0;
            for (size_t i = 0; i < rows_; ++i)
            {
                if constexpr (is_same_v<T, complex<double>>)
                {
                    colSum += abs(data_[i][j]);
                }
                else
                {
                    colSum += abs(static_cast<double>(data_[i][j]));
                }
            }
            maxColSum = max(maxColSum, colSum);
        }
        return maxColSum;
    }
    //���������׷�����AA*���������ֵ��ƽ������A*Ϊ������󣡣�������ת�þ���
    double spectralNorm() const
    {
        if (rows_ == 0 || cols_ == 0)
        {
            return 0.0;
        }

        // ���� A^H * A������ת�� * ԭ����
        Matrix<T> AH = this->conjugateTranspose();
        Matrix<T> AHA = AH * (*this);

        // ���� A^H * A ������ֵ
        auto eigenvals = AHA.eigenvalues();

        // �ҳ��������ֵ��ģ
        double maxEigenvalue = 0.0;
        for (const auto& val : eigenvals)
        {
            double absVal = abs(val);
            if (absVal > maxEigenvalue)
            {
                maxEigenvalue = absVal;
            }
        }
        // �׷������������ֵ��ƽ����
        return sqrt(maxEigenvalue);
    }
    // ����������������Ǿ���ķ������������ķ���
    double conditionNumber(const string& normType) const
    {
        if (!isSquare())
        {
            throw invalid_argument("������ֻ�Է����ж���");
        }
        try
        {
            T det = determinant();
            if (isZero(det))
            {
                return numeric_limits<double>::infinity();// �������ʽΪ�㣬���������
            }
            double normA = 0, normInv = 0;
            auto inv = this->inverse();
            if (normType == "frobenius")
            {
                normA = this->frobeniusNorm();
                normInv = inv.frobeniusNorm();
            }
            else if (normType == "row")
            {
                normA = this->rowSumNorm();
                normInv = inv.rowSumNorm();
            }
            else if (normType == "column")
            {
                normA = this->columnSumNorm();
                normInv = inv.columnSumNorm();
            }
            else if (normType == "spectral")
            {
                normA = this->spectralNorm();
                normInv = inv.spectralNorm();
            }
            else
            {
                throw invalid_argument("δ֪��������: " + normType);
            }
            return normA * normInv;
        }
        catch (const exception&)//�Գ������÷�ʽ�������б�׼�쳣��������󲻿���ȵ�
        {
            return numeric_limits<double>::infinity();//�����������
        }
    }
    //ʹ��at�������ʾ���Ԫ��
    // ��const�汾�����ؿ��޸ĵ�����
    T& at(size_t row, size_t col)
    {
        if (row >= rows_ || col >= cols_)
            throw out_of_range("��������Խ��");
        return data_[row][col];
    }
    // const�汾������ֻ������
    const T& at(size_t row, size_t col) const
    {
        if (row >= rows_ || col >= cols_)
            throw out_of_range("��������Խ��");
        return data_[row][col];
    }
    //��һ��������complex<double>����ʽ��Ϊ�ַ����������Ѻõ����������
    //ԭ����ʽ�ǣ�a��b������ʽ,ת����a+bi
    static string formatComplex(const complex<double>& c)
    {
        stringstream ss;//����һ���ַ���������
        const double EPSILON = 1e-12;
        double real = c.real();
        double imag = c.imag();
        //����߼�˵��
        bool show_real = abs(real) > EPSILON;//�Ƚ�ʵ����0���бȽ�
        bool show_imag = abs(imag) > EPSILON;//�鲿��0�Ƚ�
        if (show_real)
        {
            ss << real;
        }
        if (show_imag)//���鲿��Ϊ0�������
        {
            if (show_real)
            {
                ss << (imag > 0 ? "+" : "-");
            }
            else if (imag < 0)
            {
                ss << "-";
            }
            double abs_imag = abs(imag);
            if (abs(abs_imag - 1.0) > EPSILON)//�ж��鲿�ǲ���1���ڷ����
            {
                ss << abs_imag;
            }
            ss << "i";
        }
        if (!show_real && !show_imag)//ʵ���鲿��Ϊ0
        {
            ss << "0";
        }
        return ss.str();//���ó�Ա��������ȡ�ַ������е�ǰ�ۻ������ݣ���Ϊһ�� string ���ͷ���
    }
    //������Է�����
    //����ֵ��һ����Ԫ������ͣ�ǰ���ǽ������������ǽ������
    //������ͣ�0��ʾ�޽⣬1��ʾΨһ�⣬2��ʾ������
    //�㷨���ȴ����������Ȼ���仯Ϊ�������ʽ��
    //���ż���ϵ����������������ȣ��жϽ�����
    //���Ÿ���ϵ����������жϽ�����ͣ�
    //�����ȡ�����������ؽ������ͽ������
    pair<vector<T>, int> solveLinearSystem(const vector<T>& b) const
    {
        if (b.size() != rows_)
        {
            throw invalid_argument("������������ά�ȱ��������������ͬ");
        }
        // ����������� [A|b]
        Matrix<T> augmented(rows_, cols_ + 1);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                augmented.at(i, j) = at(i, j);
            }
            augmented.at(i, cols_) = b[i];
        }
        // ���������Ϊ�������ʽ
        Matrix<T> rref = augmented.reducedRowEchelonForm();
        // ����ϵ���������
        int rankA = rank();
        // ��������������
        int rankAb = rref.rank();
        // ����Ƿ��н⣬�����ȵıȽ�
        if (rankA < rankAb)
        {
            // ���ϵ���������С�����������ȣ����޽�
            return { vector<T>(), 0 };
        }
        // ���н��ǰ���£��жϽ�����
        int solutionType;
        if (rankA < cols_)
        {
            solutionType = 2; // ������
        }
        else
        {
            solutionType = 1; // Ψһ��
        }
        // �������������
        vector<T> solution(cols_, T(0));// ��ʼ��������Ϊ��
        // �ҳ���Ԫ�������;�����ÿһ�е�һ������Ԫ�ص�λ��
        vector<size_t> pivotCols;
        vector<size_t> pivotRows;
        size_t r = 0;// ��ǰ��
        //�ӵ�һ�п�ʼ������ÿһ�У�Ѱ�Ҳ�Ϊ0����Ԫ
        for (size_t j = 0; j < cols_ && r < rows_; ++j)
        {
            if (!isZero(rref.at(r, j)))
            {
                pivotCols.push_back(j);//��¼��Ԫ���ڵ��е�pivotCols
                pivotRows.push_back(r);//��¼��Ԫ���ڵ��е�pivotRows
                r++;
            }
        }
        // ����ÿһ����Ԫλ�ã���ȡ��Ӧ�Ľ⣬Ψһ��ʱ�����һ�о��ǽ⣡������
        //��Ԫ���ڵ�ÿһ�У����һ�о��Ƕ�Ӧδ֪���Ľ⣬��Ϊǰ���Ѿ��������������ʽ.��ԪΪ1
        for (size_t i = 0; i < pivotCols.size(); ++i)
        {
            //��Ԫ�����к�                      ��Ԫ�����кţ���һ�����һ�е�Ԫ��
            solution[pivotCols[i]] = rref.at(pivotRows[i], cols_);//�ж�ӦXi=���һ��Ԫ��
        }
        return { solution, solutionType };
    }

    // ���������Է����� AX=0 �Ļ�����ϵ
    //�㷨���Ƚ�����Ϊ�������ʽ���ҳ���Ԫ�к������У�
    //���û�����ɱ�������ֻ����⣻����Ϊÿ�����ɱ�������һ������������
	//����������������һ�����ɱ���Ϊ1���������ɱ���Ϊ0����Ԫ����ͨ���ش����õ�������

    vector<vector<T>> solveHomogeneousSystem() const
    {
        // 1. ��Ϊ�������
        Matrix<T> rref = this->reducedRowEchelonForm();

		// 2. ����Ԫ�к����ɱ����У����ɱ���������Щû����Ԫ����
        vector<size_t> pivotCols;
        vector<size_t> freeCols;
        vector<size_t> pivotRows;
        size_t r = 0;
        for (size_t j = 0; j < cols_ && r < rows_; ++j)
        {
            if (!isZero(rref.at(r, j)))
            {
                pivotCols.push_back(j);
                pivotRows.push_back(r);
                r++;
            }
        }
        // ʣ�µ��ж������ɱ������ҵ����е����ɱ���
        for (size_t j = 0; j < cols_; ++j)
        {
            if (find(pivotCols.begin(), pivotCols.end(), j) == pivotCols.end())
            //find(��ʼ������, ����������, Ҫ���ҵ�ֵ) �����û�ҵ������� pivotCols.end()      ����û�ҵ�
                freeCols.push_back(j);//��ӵ����ɱ�������
        }

        // 3. ���û�����ɱ�����ֻ�����
        if (freeCols.empty())
        {
            return {};
        }
        // 4. Ϊÿ�����ɱ�������һ������������
        vector<vector<T>> basisVectors;//������ϵ
		for (size_t k = 0; k < freeCols.size(); ++k)// ��ÿ�����ɱ�������һ������������
        {
			vector<T> sol(cols_, T(0));// ��ʼ��������Ϊ��
            sol[freeCols[k]] = T(1); // ��ǰ���ɱ�����Ϊ1
            // �ش�����Ԫ����
			for (size_t i = 0; i < pivotCols.size(); ++i)//������Ԫ��
            {
                size_t row = pivotRows[i];
                size_t col = pivotCols[i];
                T sum = T(0);
                // ע�⣺�������ɱ�����Ҫ���룬��ʹ��0
                for (size_t j = 0; j < freeCols.size(); ++j)
                {
                    sum += rref.at(row, freeCols[j]) * sol[freeCols[j]];
                    //������Ԫ�����е��������ɱ����Ĺ���
                }
				sol[col] = -sum;// ������Ԫ������ֵ
            }
			basisVectors.push_back(sol);// ��������������ӵ�����У��������ÿ������
        }
        return basisVectors;
    }

	// ����������Է����� Ax=b ��ͨ�⣬���൱�����ؽ����η�����Ļ�����ϵ
    pair<vector<T>, vector<vector<T>>> solveLinearSystemComplete(const vector<T>& b) const
    {
        pair<vector<T>, int> result = solveLinearSystem(b);
        vector<T> particularSolution = result.first;
        int solutionType = result.second;
        // ����޽⣬ֱ�ӷ���
        if (solutionType == 0)
        {
            return { particularSolution, {} };
        }
        // �����Ψһ�⣬�����ؽ�Ϳջ�����ϵ
        if (solutionType == 1)
        {
            return { particularSolution, {} };
        }
        // ����������⣬����η�����Ļ�����ϵ
        vector<vector<T>> basisVectors = solveHomogeneousSystem();
        return { particularSolution, basisVectors };
    }
	// ����ָ�����㷽�������������������΢�ַ�����
    Matrix<T> matrixExponential(double t) const
    {
        if (!isSquare())
        {
            throw invalid_argument("����ָ��ֻ�Է����ж���");
        }
        // ʹ��̩�ռ����������ָ�� e^(At)
        Matrix<T> result = Matrix<T>::identity(rows_);
        Matrix<T> term = Matrix<T>::identity(rows_);
        double factorial = 1.0;
        // ����ǰ100��̩�ռ������������㹻��ȷ�������кܴ�����
        for (int i = 1; i <= 100; ++i)
        {
            term = term * (*this) * (t / i);  // A^i * t^i / i!
            factorial *= i;//����׳ˣ�����̫��Ҫ������ɾ��
            result = result + term;
            // ������÷ǳ�С��������ǰ��ֹ
            if (term.frobeniusNorm() < EPSILON)
            {
                break;
            }
        }
        return result;
    }
	// ���һ���������΢�ַ����� dx/dt = Ax, x(0) = x0��ʹ�ù�ʽ���� x(t) = e^(At) * x0
    vector<T> solveLinearODE(const vector<T>& x0, double t) const
    {
        if (!isSquare()) 
        {
            throw invalid_argument("ϵ����������Ƿ���");
        }
        if (x0.size() != rows_)
        {
            throw invalid_argument("��ʼֵ����ά�ȱ��������������ͬ");
        }
        // �������ָ�� e^(At)
        Matrix<T> expAt = matrixExponential(t);
        // ���� x(t) = e^(At) * x0
        vector<T> result(rows_, T(0));
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result[i] += expAt.at(i, j) * x0[j];
            }
        }
        return result;
    }
	// ������λ������̩�ռ����������ָ����ʱ���õ�����һ����ǵ�λ����
    static Matrix<T> identity(size_t size)
    {
        Matrix<T> mat(size, size, T(0));
        for (size_t i = 0; i < size; ++i)
        {
            if constexpr (is_same_v<T, complex<double>>)
            {
                mat.data_[i][i] = complex<double>(1.0, 0.0);
            }
            else
            {
                mat.data_[i][i] = T(1);
            }
        }
        return mat;
    }
	// �������ת����������ԭ�����͵ľ���ת��Ϊ�������͵ľ��󣬼��ݲ�ͬ���͵ľ������㣬ʵ������������ܳ��ָ���
    //���԰� Matrix<double> ת��Ϊ Matrix<complex<double>>�����߷�������
    template<typename U>
    Matrix<U> cast() const
    {
        Matrix<U> result(rows_, cols_);//
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                // ���ڸ���������Ҫ���⴦��
                if constexpr (is_same_v<T, complex<double>>)
                {
                    if constexpr (is_same_v<U, double>)
                    {
						result.at(i, j) = data_[i][j].real();// ֻȡʵ��
                    }
                    else
                    {
                        result.at(i, j) = static_cast<U>(data_[i][j].real());
                    }
                }
                else
                {
                    result.at(i, j) = static_cast<U>(data_[i][j]);
                }
            }
        }
        return result;
    }
private:
    // �жϾ����Ƿ�Ϊ����
    bool isSquare() const { return rows_ == cols_; }
    // ���ά���Ƿ���ͬ�����ж�Ҫ��ͬ��
    void checkSameDimensions(const Matrix& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw invalid_argument("ά�Ȳ�ƥ��");
    }
    //�ж���ֵ�Ƿ�Ϊ��
    bool isZero(const T& val) const
    {
        if constexpr (is_same_v<T, complex<double>>)
        {
            return abs(val) < EPSILON;
        }
        else if constexpr (is_same_v<T, double> || is_same_v<T, float>)
        {
            return abs(val) < EPSILON;
        }
        else
        {
            return val == 0;
        }
    }
	//�����������ʽ��ʱ��ʹ�ã����Ӿ���
    Matrix<T> subMatrix(size_t excludeRow, size_t excludeCol) const
    {
        if (rows_ <= 1 || cols_ <= 1)
        {
            throw std::invalid_argument("�޷���1x1���С�ľ���ȡ�Ӿ���");
        }
        if (excludeRow >= rows_ || excludeCol >= cols_)
        {
            throw std::out_of_range("subMatrix����Խ��");
        }
        Matrix<T> result(rows_ - 1, cols_ - 1);
        for (size_t i = 0, r = 0; i < rows_; ++i)
        {
            if (i == excludeRow)
            {
				continue;// ����Ҫ�ų�����
            }
            for (size_t j = 0, c = 0; j < cols_; ++j)
            {
                if (j == excludeCol)
                {
                    continue;//����Ҫ�ų�����
                }
                result.data_[r][c++] = data_[i][j];
            }
            r++;
        }
        return result;
    }
    //static���ã���Щ���ߺ�����������Ҫ�����ض�����ķǾ�̬��Ա����������ֻ��������ľ��������
    //Ѱ����Ԫ�����ߺ�����������ʽ������������ȵ�ʱ������Ѱ�ҵ�ǰ�й涨��Χ�ڵ�һ������Ԫ�ص�ֵ
    //������һ����Ѱ�ҵ�һ����Ϊ���Ԫ�ص���
    static size_t findPivot(Matrix& mat, size_t col, size_t startRow)
    {
        size_t pivot = startRow;
        double maxAbs = 0;
        for (size_t i = startRow; i < mat.rows_; ++i)
        {
            double absVal = abs(mat.data_[i][col]);
            if (absVal > maxAbs)
            {
                maxAbs = absVal;
                pivot = i;
            }
        }
        if (maxAbs < EPSILON) return mat.rows_; // ȫ��
        return pivot;
    }
    //�����У�Ҳ��һ�����ߺ��������ڰѵ�ǰ�к���Ԫ�����н��н���
    static void swapRows(Matrix& mat, size_t a, size_t b)
    {
        swap(mat.data_[a], mat.data_[b]);//����algorithm�������swap�������ڽ���
    }
    //���ߺ���������Ԫ��pivot��λ�õ�ֵ��Ϊ1�����ں�������ͻ���Ϊ������;���
    //�㷨���ҵ������У�Ȼ��ͬ��
    static void normalizeRow(Matrix& mat, size_t row, size_t col)
    {
        T pivot = mat.data_[row][col];//�ҵ���Ԫ���ڵ��к���
        if (mat.isZero(pivot))//�����ԪΪ�㣬����Ҫ��Ϊ1
        {
            return;
        }
        for (size_t j = col; j < mat.cols_; ++j)
        {
            mat.data_[row][j] /= pivot;
        }
    }
    //���ߺ�������ȥ��Ԫ�·����У�ר��������������
    static void eliminateRows(Matrix& mat, size_t row, size_t col)
    {
        for (size_t i = row + 1; i < mat.rows_; ++i)
        {
            T factor = mat.data_[i][col];//��Ϊ��ԪΪ1������ֱ�ӳ��Ե�ǰԪ�صĸ�����Ȼ���ӦԪ�����
            for (size_t j = col; j < mat.cols_; ++j)
            {
                mat.data_[i][j] -= factor * mat.data_[row][j];
            }
        }
    }
    //���ߺ������Ὣ��Ԫ���ϳ�����Ԫ����������з���Ԫ�ض���Ϊ0��ר��������������;���
    static void eliminateAllRows(Matrix& mat, size_t row, size_t col) {
        for (size_t i = 0; i < mat.rows_; ++i) {
            if (i != row && !mat.isZero(mat.data_[i][col])) {
                T factor = mat.data_[i][col];
                for (size_t j = col; j < mat.cols_; ++j) {
                    mat.data_[i][j] -= factor * mat.data_[row][j];
                }
            }
        }
    }
    //���ߺ������ڲ�����Ԫ����1��������������Ԫ�Ĳ������ڼ�������ʽ��ֵ��ʱ��ʹ��
    //���Ҵ���Ĳ���Ҳ��һ��
    static void eliminateLower(Matrix& mat, size_t row)
    {
        for (size_t i = row + 1; i < mat.rows_; ++i)
        {
            T factor = mat.data_[i][row] / mat.data_[row][row];
            for (size_t j = row; j < mat.cols_; ++j)
            {
                mat.data_[i][j] -= factor * mat.data_[row][j];
            }
        }
    }
};
// ��������ר������ʽʵ�֣�ģ���ػ���
// ��Ϊ��������������ʽ��ֵ��ʱ���ʵ��������һ������
// ������Ҫ��ģ�Ƚϣ����ߺ�������ֱ��ʹ��ʹ�������ػ�
template<>//ģ���ػ�����
complex<double> Matrix<complex<double>>::determinant() const
{
    if (!isSquare()) 
    {
        throw invalid_argument("�Ƿ����޷���������ʽ");
    }
    Matrix<complex<double>> mat = *this;
	complex<double> det = 1.0;// ��ʼ������ʽΪ1
	int swapCount = 0;// ��¼�н�������
	size_t n = mat.rows_;// ��ȡ���������
    for (size_t col = 0; col < n; ++col) 
    {
        // ѡ��Ԫ��ģ���
        size_t pivot = col;
        double maxAbs = abs(mat.data_[col][col]);
        for (size_t i = col + 1; i < n; ++i) 
        {
            double valAbs = abs(mat.data_[i][col]);
            if (valAbs > maxAbs) 
            {
                maxAbs = valAbs;
                pivot = i;
            }
        }
        if (maxAbs < EPSILON) return complex<double>(0.0, 0.0); // ����ʽΪ0
        if (pivot != col) //�н���
        {
            swap(mat.data_[col], mat.data_[pivot]);
            swapCount++;
        }
        det *= mat.data_[col][col];
        // ��Ԫ
        for (size_t i = col + 1; i < n; ++i) 
        {
            complex<double> factor = mat.data_[i][col] / mat.data_[col][col];
            for (size_t j = col; j < n; ++j)
            {
                mat.data_[i][j] -= factor * mat.data_[col][j];
            }
        }
    }
	if (swapCount % 2)// ��������еĴ�����������������ʽȡ��
    {
        det = -det;
    }
    // ��ʵ�����鲿��С������
    if (abs(det.real()) < EPSILON)
    {
        det.real(0.0);
    }
    if (abs(det.imag()) < EPSILON)
    {
        det.imag(0.0);
    }
    return det;
}
// ʵ����������ֵ��֧�ָ�����ֵ��2x2��ֱ�ӽ���������ʽ��
//�㷨��ʹ��QR�㷨�����������ֵ��ֱ�����������������ʽ
// ������������ͨ������ maxIterations ���ƣ�Ĭ��100��
// ������������ֵ����ʹ����QR�ֽ�͵���������
// ͨ��Gram-Schmidt���������̹�����������Q�������Ǿ���R��
// Ȼ��������¾���A��ֱ������Ϊֹ
// �����ȡ����ֵ��֧��2x2�鸴����ֵ

template<>
vector<complex<double>> Matrix<double>::eigenvalues(int maxIterations) const //�ø��������Ž��
{
    // ����Ƿ�Ϊ����
    size_t n = this->getRows();
    if (n != this->getCols()) throw invalid_argument("����ֵֻ�ܼ��㷽��");
    // ��������
    vector<vector<double>> A(n, vector<double>(n));//����A�����п���
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            A[i][j] = this->at(i, j);
    double tolerance = 1e-10;// �������̶�
    for (int iter = 0; iter < maxIterations; ++iter)//460�е�ʱ������������Ĭ��ֵ
    {
        // Gram-Schmidt QR�ֽ�
		vector<vector<double>> Q(n, vector<double>(n, 0));// ��������Q���洢���ǡ�e1,e2,e3....��
		vector<vector<double>> R(n, vector<double>(n, 0));// �����Ǿ���R
        //�����ȰѾ���д������������ʽ
        for (size_t j = 0; j < n; ++j)//����ÿһ��
        {   
			vector<double> v(n);// ���ڴ洢��ǰ������
            for (size_t i = 0; i < n; ++i)
            {
				v[i] = A[i][j];// ȡ��ǰ������,����Gram-Schmidt����������j��
            }
            for (size_t k = 0; k < j; ++k) 
            {
				double proj = 0, normQ = 0;// ����ͶӰ��Q�ķ�����ƽ��
                for (size_t i = 0; i < n; ++i)
                {
					proj += v[i] * Q[i][k];// ����ͶӰ���൱������e1*e2
					normQ += Q[i][k] * Q[i][k];// ����Q�ķ�����ƽ��������e2��ģ��ƽ��
                }
                if (normQ < tolerance)
                {
					normQ = tolerance;// ��ֹ������
                }
				proj = proj / normQ;// ����ͶӰϵ����e1��ת�ó���ak
                for (size_t i = 0; i < n; ++i)
					v[i] -= proj * Q[i][k];// �ӵ�ǰ�������м�ȥͶӰ���֣��õ��������������
				    R[k][j] = proj;// �洢ͶӰϵ��
            }
            double norm_val = 0.0;
            //��e1��λ����֮��Ҳһ��
            for (size_t i = 0; i < n; ++i) norm_val += v[i] * v[i];//����ģ����ƽ��
            norm_val = sqrt(norm_val);// ���㵱ǰ��������ģ��
            if (norm_val < tolerance)
            {
                v[j] = 1;
                norm_val = 1.0;
            }
            for (size_t i = 0; i < n; ++i)
                Q[i][j] = v[i] / norm_val;//Q�����洢�ľ��൱���ǡ�e1.e2,e3....��
                R[j][j] = norm_val;
        }
        // A = R * Q
        vector<vector<double>> newA(n, vector<double>(n, 0));// �µľ���A
        // �����¾���A = R * Q
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < n; ++k)
                    newA[i][j] += R[i][k] * Q[k][j];
        // ����Ƿ����������
        bool converged = true;
        for (size_t i = 1; i < n; ++i)
            if (fabs(newA[i][i - 1]) > tolerance)
            {
                converged = false;
                break;
            }
        A = newA;
        if (converged) break;
    }
    // ��ȡ����ֵ��֧��2x2�鸴����ֵ��
    vector<complex<double>> eigenvals;
    size_t i = 0;
    while (i < n) 
    {
        if (i + 1 < n && fabs(A[i + 1][i]) > 1e-6) {
            // 2x2�飬�� ��^2 - tr*�� + det = 0
            double a = 1.0;
            double b = -(A[i][i] + A[i + 1][i + 1]);
            double c = A[i][i] * A[i + 1][i + 1] - A[i][i + 1] * A[i + 1][i];
            double delta = b * b - 4 * a * c;
            if (delta >= 0)
            {
                double l1 = (-b + sqrt(delta)) / (2 * a);
                double l2 = (-b - sqrt(delta)) / (2 * a);
                eigenvals.push_back(complex<double>(l1, 0));
                eigenvals.push_back(complex<double>(l2, 0));
            }
            else
            {
                double real = -b / (2 * a);
                double imag = sqrt(-delta) / (2 * a);
                eigenvals.push_back(complex<double>(real, imag));
                eigenvals.push_back(complex<double>(real, -imag));
            }
            i += 2;
        }
        else 
        {
            eigenvals.push_back(complex<double>(A[i][i], 0));
            i++;
        }
    }
    return eigenvals;
}
// ������ʵ��������������
template<typename T>
ostream& operator<<(ostream& os, const Matrix<T>& mat)
{
    os << mat.rows_ << " " << mat.cols_ << endl;
    for (const auto& row : mat.data_)
    {
        for (const T& elem : row)
        {
            os << elem << " ";
        }
        os << endl;
    }
    return os;
}
template<typename T>
istream& operator>>(istream& is, Matrix<T>& mat)
{
    is >> mat.rows_ >> mat.cols_;//�������к���
    //vector����ĺ���
    mat.data_.resize(mat.rows_, vector<T>(mat.cols_));//ȷ������Ĵ洢�ռ����ָ��������������һ��
    // ������������ֱ���������з�Ϊֹ�������ַ�����������ջ�����������ǰ��ʣ�����ݡ�
    is.ignore(numeric_limits<streamsize>::max(), '\n');
    for (size_t i = 0; i < mat.rows_; ++i)
    {
        string line;
        getline(is, line);
        // �����ȡ�����У�����
        if (line.empty() && i < mat.rows_)
        {
            i--;
            continue;
        }
        istringstream iss(line);//ת��������������� iss >> ����
        for (size_t j = 0; j < mat.cols_; ++j)
        {
            T value;
            if (!(iss >> value))
            {
                //����ʹ��to_string������ֵ����ת��Ϊ�ַ�����
                throw runtime_error("��Ч�ľ���Ԫ�ظ�ʽ���� " + to_string(i + 1) + " �� " + to_string(j + 1));
            }
            mat.at(i, j) = value;  // ʹ��at()����
        }
    }
    return is;
}
// �ػ��������������������������
template<>
istream& operator>>(istream& is, Matrix<complex<double>>& mat)
{
    // ��ȡ������
    if (!(is >> mat.rows_ >> mat.cols_))
    {
        throw runtime_error("��Ч����������ʽ");
    }
    mat.data_.resize(mat.rows_, vector<complex<double>>(mat.cols_));

    is.ignore(numeric_limits<streamsize>::max(), '\n');
    for (size_t i = 0; i < mat.rows_; ++i) {
        string line;
        getline(is, line);
        // �����ȡ�����У�����
        if (line.empty() && i < mat.rows_)
        {
            i--;
            continue;
        }
        istringstream iss(line);
        for (size_t j = 0; j < mat.cols_; ++j)
        {
            string input;
            if (!(iss >> input))
            {
                throw runtime_error("��Ч�ľ���Ԫ�ظ�ʽ���� " + to_string(i + 1) + " �� " + to_string(j + 1));
            }
            // ����(real,imag)��ʽ�ĸ���
            if (input.front() == '(' && input.back() == ')')
            {
                // ȥ������
                input = input.substr(1, input.size() - 2);//��ʼλ�ã�Ҫ��ȡ�ĳ���
                // ���Ҷ���λ��
                size_t comma_pos = input.find(',');
                if (comma_pos != string::npos)//��������
                {
                    double real = stod(input.substr(0, comma_pos));//stdod���ַ���ת��Ϊdouble
                    double imag = stod(input.substr(comma_pos + 1));
                    mat.at(i, j) = complex<double>(real, imag);  // ʹ��at()����
                    continue;
                }
            }
            // �������(real,imag)��ʽ������ʹ��a+bi��ʽ����
            complex<double> value(0, 0);
            try {
                // ȥ�������ַ����п��ܴ��ڵĿո�erase(first, last); // ɾ��[first, last)����������ַ�
                input.erase(remove_if(input.begin(), input.end(), ::isspace), input.end());
                // ����Ƿ�Ϊ��������ʽ (i, -i, 3i, -3i)
                if (input == "i")
                {
                    value = complex<double>(0, 1);
                }
                else if (input == "-i")
                {
                    value = complex<double>(0, -1);
                }
                else if (input.back() == 'i')
                {
                    // ����������ʽ�Ĵ����� (�� 3i, -3i)
                    string imag_part = input.substr(0, input.size() - 1);
                    if (imag_part.empty() || imag_part == "+")
                    {
                        value = complex<double>(0, 1);
                    }
                    else if (imag_part == "-")
                    {
                        value = complex<double>(0, -1);
                    }
                    else
                    {
                        value = complex<double>(0, stod(imag_part));
                    }
                }
                // ������мӺŵĸ��� (a+bi)
                if (input.find('+') != string::npos && input.find('i') != string::npos)
                {
                    size_t plus_pos = input.find('+');
                    double real_part = stod(input.substr(0, plus_pos));//��ȡʵ��
                    string imag_str = input.substr(plus_pos + 1);//��ȡ�鲿
                    double imag_part = 1.0;// Ĭ���鲿Ϊ1
                    if (imag_str != "i")
                    {
                        imag_part = stod(imag_str.substr(0, imag_str.size() - 1));
                    }
                    value = complex<double>(real_part, imag_part);
                }
                // ������м��ŵĸ��� (a-bi)
                else if (input.find('-', 1) != string::npos && input.find('i') != string::npos)
                {
                    size_t minus_pos = input.find('-', 1);
                    double real_part = stod(input.substr(0, minus_pos));
                    string imag_str = input.substr(minus_pos + 1);
                    double imag_part = 1.0;
                    if (imag_str != "i") {
                        imag_part = stod(imag_str.substr(0, imag_str.size() - 1));
                    }
                    value = complex<double>(real_part, -imag_part);
                }
                // ����������ʽ (i, -i, 3i, -3i)
                else if (input.back() == 'i')
                {
                    // ����������ʽ�Ĵ����� (�� 3i, -3i)
                    string imag_part = input.substr(0, input.size() - 1);
                    if (imag_part.empty() || imag_part == "+")
                    {
                        value = complex<double>(0, 1);
                    }
                    else if (imag_part == "-")
                    {
                        value = complex<double>(0, -1);
                    }
                    else
                    {
                        value = complex<double>(0, stod(imag_part));
                    }
                }
                // ����ʵ��
                else
                {
                    value = complex<double>(stod(input), 0);
                }
            }
            catch (const exception& e)
            {
                throw runtime_error("��Ч�ĸ�����ʽ: " + input + " (" + e.what() + ")");
            }

            mat.at(i, j) = value;  // ʹ��at()����
        }
    }
    return is;
}
template<typename T>
ostream& operator<<(ostream& os, const complex<double>& c)
{
    double real = c.real();
    double imag = c.imag();
    const double EPSILON = 1e-12;
    // ����ʵ����ʾ
    bool show_real = abs(real) > EPSILON;
    // �����鲿��ʾ
    bool show_imag = abs(imag) > EPSILON;
    if (show_real) os << real;
    if (show_imag)
    {
        // �������
        if (show_real)
        {
            os << (imag > 0 ? "+" : "-");
        }
        else if (imag < 0) {
            os << "-";
        }
        // �����鲿��ֵ
        double abs_imag = abs(imag);
        if (abs(abs_imag - 1.0) > EPSILON)
        {
            os << abs_imag;
        }
        os << "i";
    }
    // ����ȫ�����
    if (!show_real && !show_imag) os << "0";
    return os;
}
// �������������ʵ��
// ��������QR�ֽ�+����ֵ��Gram-Schmidt�棩
template<>
vector<complex<double>> Matrix<complex<double>>::eigenvalues(int maxIterations) const
{
    size_t n = this->getRows();
    if (n != this->getCols()) throw invalid_argument("����ֵֻ�ܼ��㷽��");

    // ��������
    vector<vector<complex<double>>> A(n, vector<complex<double>>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            A[i][j] = this->at(i, j);

    double tolerance = 1e-10;
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // Gram-Schmidt QR�ֽ�
        vector<vector<complex<double>>> Q(n, vector<complex<double>>(n, 0));
        vector<vector<complex<double>>> R(n, vector<complex<double>>(n, 0));
        for (size_t j = 0; j < n; ++j)
        {
            vector<complex<double>> v(n);
            for (size_t i = 0; i < n; ++i) v[i] = A[i][j];
            for (size_t k = 0; k < j; ++k)
            {
                complex<double> proj = 0;
                double normQ = 0.0;
                for (size_t i = 0; i < n; ++i)
                {
                    proj += v[i] * conj(Q[i][k]);
                    normQ += norm(Q[i][k]);
                }
                if (normQ < tolerance) normQ = tolerance;
                proj = proj / complex<double>(normQ, 0);
                for (size_t i = 0; i < n; ++i)
                    v[i] -= proj * Q[i][k];
                R[k][j] = proj;
            }
            double norm_val = 0.0;
            for (size_t i = 0; i < n; ++i) norm_val += norm(v[i]);
            norm_val = sqrt(norm_val);
            if (norm_val < tolerance)
            {
                v[j] = 1;
                norm_val = 1.0;
            }
            for (size_t i = 0; i < n; ++i)
                Q[i][j] = v[i] / complex<double>(norm_val, 0);
            R[j][j] = complex<double>(norm_val, 0);
        }
        // A = R * Q
        vector<vector<complex<double>>> newA(n, vector<complex<double>>(n, 0));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < n; ++k)
                    newA[i][j] += R[i][k] * Q[k][j];
        // �������
        bool converged = true;
        for (size_t i = 1; i < n; ++i)
            if (abs(newA[i][i - 1]) > tolerance)
            {
                converged = false;
                break;
            }
        A = newA;
        if (converged) break;
    }
    // ��ȡ�Խ���Ϊ����ֵ
    vector<complex<double>> eigenvals(n);
    for (size_t i = 0; i < n; ++i)
        eigenvals[i] = A[i][i];
    return eigenvals;
}
void showMenu()
{
    while (true)
    {
        system("cls"); // ����
        cout << "�����������\n";
        cout << "1. �����¾�����������\n";
        cout << "2. ���ļ����ؾ���\n";
        cout << "3. �˳�����\n";
        cout << "��ѡ�������1-3��: ";
        int inputType;
        cin >> inputType;
        if (inputType == 3) break;
        MatrixBase* mat = nullptr;// �������ָ�룬���ڴ洢��ͬ���͵ľ���
        bool isComplex = false;//
        if (inputType == 1)
        {
            system("cls");
            cout << "��ѡ��������ͣ�\n";
            cout << "1. ʵ������\n";
            cout << "2. ��������\n";
            cout << "��ѡ��1-2����";
            int type;
            cin >> type;
            system("cls");
            if (type == 1)
            {
                auto* m = new Matrix<double>();
                cout << "����������� �У�����ÿ������Ԫ�أ���\n";
                cin >> *m;
                mat = m;
            }
            else
            {
                auto* m = new Matrix<complex<double>>();
                cout << "�����븴�������� �У�����ÿ������Ԫ�أ�������1+2i����\n";
                cin >> *m;
                mat = m;
                isComplex = true;
            }
        }
        else if (inputType == 2)
        {
            system("cls");
            cout << "�������ļ�����";
            string filename;
            cin >> filename;
            cout << "��ѡ��������ͣ�\n";
            cout << "1. ʵ������\n";
            cout << "2. ��������\n";
            cout << "��ѡ��1-2����";
            int type;
            cin >> type;
            system("cls");
            if (type == 1)
            {
                auto* m = new Matrix<double>(Matrix<double>::loadFromFile(filename));
                mat = m;
            }
            else
            {
                auto* m = new Matrix<complex<double>>(Matrix<complex<double>>::loadFromFile(filename));
                mat = m;
                isComplex = true;
            }
        }
        else
        {
            continue;
        }

        // ����˵�
        while (true)
        {
            system("cls");
            cout << "\n��������ѡ�\n";
            cout << "1. ����ת��\n";
            cout << "2. ����ӷ�\n";
            cout << "3. �������\n";
            cout << "4. ����˷�\n";
            cout << "5. ��������\n";
            cout << "6. ����������;���\n";
            cout << "7. ������������\n";
            cout << "8. ���������\n";
            cout << "9. �������������\n";
            cout << "10. ���㷽������ֵ\n";
            cout << "11. ���㷽������ʽ\n";
            cout << "12. ����������\n";
            cout << "13. ����������\n";
            cout << "14. ���һ���������΢�ַ����� dx/dt=Ax\n";
            cout << "15. ������Է����� AX=b\n";
            cout << "16. �������˵�\n";
            cout << "��ѡ�������1-16����";
            int op;
            cin >> op;
            system("cls");
            if (op == 16) break;
            try
            {
                // ����ʾԭ����
                cout << "ԭ����Ϊ��\n";
                if (isComplex)
                    static_cast<Matrix<complex<double>>*>(mat)->print();
                else
                    static_cast<Matrix<double>*>(mat)->print();
                cout << endl;

                if (op == 1)
                {
                    cout << "ת�ý����\n";
                    if (isComplex)
                        static_cast<Matrix<complex<double>>*>(mat)->transpose().print();
                    else
                        static_cast<Matrix<double>*>(mat)->transpose().print();
                }
                else if (op == 2 || op == 3 || op == 4)
                {
                    cout << "��������һ�������� �У�����ÿ������Ԫ�أ���\n";
                    if (isComplex)
                    {
                        Matrix<complex<double>> m2;
                        cin >> m2;
                        cout << "\nԭ����Ϊ��\n";
                        static_cast<Matrix<complex<double>>*>(mat)->print();
                        cout << "\n��һ������Ϊ��\n";
                        m2.print();
                        cout << "\n���Ϊ��\n";
                        if (op == 2)
                            (static_cast<Matrix<complex<double>>*>(mat)->operator+(m2)).print();
                        else if (op == 3)
                            (static_cast<Matrix<complex<double>>*>(mat)->operator-(m2)).print();
                        else
                            (static_cast<Matrix<complex<double>>*>(mat)->operator*(m2)).print();
                    }
                    else
                    {
                        Matrix<double> m2;
                        cin >> m2;
                        cout << "\nԭ����Ϊ��\n";
                        static_cast<Matrix<double>*>(mat)->print();
                        cout << "\n��һ������Ϊ��\n";
                        m2.print();
                        cout << "\n���Ϊ��\n";
                        if (op == 2)
                            (static_cast<Matrix<double>*>(mat)->operator+(m2)).print();
                        else if (op == 3)
                            (static_cast<Matrix<double>*>(mat)->operator-(m2)).print();
                        else
                            (static_cast<Matrix<double>*>(mat)->operator*(m2)).print();
                    }
                }
                else if (op == 5)
                {
                    cout << "�������Ϊ��";
                    if (isComplex)
                        cout << static_cast<Matrix<complex<double>>*>(mat)->rank() << endl;
                    else
                        cout << static_cast<Matrix<double>*>(mat)->rank() << endl;
                }
                else if (op == 6)
                {
                    cout << "������;���Ϊ��\n";
                    if (isComplex)
                        static_cast<Matrix<complex<double>>*>(mat)->reducedRowEchelonForm().print();
                    else
                        static_cast<Matrix<double>*>(mat)->reducedRowEchelonForm().print();
                }
                else if (op == 7)
                {
                    cout << "�������������Ⱥ�Ԫ�أ�\n";
                    size_t n;
                    cin >> n;
                    if (isComplex)
                    {
                        vector<complex<double>> v(n);
                        for (size_t i = 0; i < n; ++i)
                        {
                            string s;
                            cin >> s;
                            v[i] = complex<double>(stod(s), 0);
                        }
                        cout << "������pֵ����1,2����";
                        double p;
                        cin >> p;
                        cout << "����Ϊ��" << Matrix<complex<double>>::vectorNorm(v, p) << endl;
                    }
                    else
                    {
                        vector<double> v(n);
                        for (size_t i = 0; i < n; ++i) cin >> v[i];
                        cout << "������pֵ����1,2����";
                        double p;
                        cin >> p;
                        cout << "����Ϊ��" << Matrix<double>::vectorNorm(v, p) << endl;
                    }
                }
                else if (op == 8)
                {
                    if (isComplex)
                    {
                        auto* m = static_cast<Matrix<complex<double>>*>(mat);
                        cout << "Frobenius����: " << m->frobeniusNorm() << endl;
                        cout << "�кͷ���: " << m->rowSumNorm() << endl;
                        cout << "�кͷ���: " << m->columnSumNorm() << endl;
                        cout << "�׷���: " << m->spectralNorm() << endl;
                    }
                    else
                    {
                        auto* m = static_cast<Matrix<double>*>(mat);
                        cout << "Frobenius����: " << m->frobeniusNorm() << endl;
                        cout << "�кͷ���: " << m->rowSumNorm() << endl;
                        cout << "�кͷ���: " << m->columnSumNorm() << endl;
                        cout << "�׷���: " << m->spectralNorm() << endl;
                    }
                }
                else if (op == 9)
                {
                    cout << "��ѡ�������ͣ�frobenius/row/column/spectral����";
                    string normType;
                    cin >> normType;
                    if (isComplex)
                        cout << "������: " << static_cast<Matrix<complex<double>>*>(mat)->conditionNumber(normType) << endl;
                    else
                        cout << "������: " << static_cast<Matrix<double>*>(mat)->conditionNumber(normType) << endl;
                }
                else if (op == 10)
                {
                    cout << "����ֵΪ��\n";
                    if (isComplex)
                    {
                        auto eig = static_cast<Matrix<complex<double>>*>(mat)->eigenvalues();
                        for (const auto& v : eig) cout << Matrix<complex<double>>::formatComplex(v) << "\t";
                        cout << endl;
                    }
                    else
                    {
                        auto eig = static_cast<Matrix<double>*>(mat)->eigenvalues();
                        for (const auto& v : eig) cout << v << "\t";
                        cout << endl;
                    }
                }
                else if (op == 11)
                {
                    if (isComplex)
                        cout << "����ʽ: " << Matrix<complex<double>>::formatComplex(static_cast<Matrix<complex<double>>*>(mat)->determinant()) << endl;
                    else
                        cout << "����ʽ: " << static_cast<Matrix<double>*>(mat)->determinant() << endl;
                }
                else if (op == 12)
                {
                    cout << "�����Ϊ��\n";
                    if (isComplex)
                    {
                        static_cast<Matrix<complex<double>>*>(mat)->inverse().print();
                        auto& A = *static_cast<Matrix<complex<double>>*>(mat);
                        auto inv = A.inverse();
                        auto prod = A * inv;
                        prod.print(); // ����Ƿ�Ϊ��λ����
                    }
                    else
                        static_cast<Matrix<double>*>(mat)->inverse().print();
                }
                else if (op == 13)
                {
                    cout << "�������Ϊ��\n";
                    if (isComplex)
                        static_cast<Matrix<complex<double>>*>(mat)->adjugate().print();
                    else
                        static_cast<Matrix<double>*>(mat)->adjugate().print();
                }
                else if (op == 14)
                {
                    cout << "�������ʼ����x0��ÿ��Ԫ���ÿո�ָ�����\n";
                    size_t n = mat->getRows();
                    if (isComplex)
                    {
                        vector<complex<double>> x0(n);
                        for (size_t i = 0; i < n; ++i)
                        {
                            string s;
                            cin >> s;
                            x0[i] = complex<double>(stod(s), 0);
                        }
                        cout << "������t��ֵ: ";
                        double t;
                        cin >> t;
                        auto xt = static_cast<Matrix<complex<double>>*>(mat)->solveLinearODE(x0, t);
                        cout << "x(t): ";
                        for (const auto& v : xt) cout << Matrix<complex<double>>::formatComplex(v) << "\t";
                        cout << endl;
                    }
                    else
                    {
                        vector<double> x0(n);
                        for (size_t i = 0; i < n; ++i) cin >> x0[i];
                        cout << "������t��ֵ: ";
                        double t;
                        cin >> t;
                        auto xt = static_cast<Matrix<double>*>(mat)->solveLinearODE(x0, t);
                        cout << "x(t): ";
                        for (const auto& v : xt) cout << v << "\t";
                        cout << endl;
                    }
                }
                else if (op == 15)
                {
                    cout << "�����볣������b��ÿ��Ԫ���ÿո�ָ�����\n";
                    size_t n = mat->getRows();
                    if (isComplex)
                    {
                        vector<complex<double>> b(n);
                        for (size_t i = 0; i < n; ++i)
                        {
                            string s;
                            cin >> s;
                            b[i] = complex<double>(stod(s), 0);
                        }
                        auto [particular, basis] = static_cast<Matrix<complex<double>>*>(mat)->solveLinearSystemComplete(b);
                        if (particular.empty() && basis.empty())
                        {
                            cout << "���Է������޽⡣" << endl;
                        }
                        else
                        {
                            cout << "�ؽ�: ";
                            for (const auto& v : particular) cout << Matrix<complex<double>>::formatComplex(v) << "\t";
                            cout << endl;
                            if (!basis.empty())
                            {
                                cout << "������ϵ:" << endl;
                                for (const auto& vec : basis)
                                {
                                    for (const auto& v : vec) cout << Matrix<complex<double>>::formatComplex(v) << "\t";
                                    cout << endl;
                                }
                            }
                        }
                    }
                    else
                    {
                        vector<double> b(n);
                        for (size_t i = 0; i < n; ++i) cin >> b[i];
                        auto [particular, basis] = static_cast<Matrix<double>*>(mat)->solveLinearSystemComplete(b);
                        if (particular.empty() && basis.empty())
                        {
                            cout << "���Է������޽⡣" << endl;
                        }
                        else
                        {
                            cout << "�ؽ�: ";
                            for (const auto& v : particular) cout << v << "\t";
                            cout << endl;
                            if (!basis.empty())
                            {
                                cout << "������ϵ:" << endl;
                                for (const auto& vec : basis)
                                {
                                    for (const auto& v : vec) cout << v << "\t";
                                    cout << endl;
                                }
                            }
                        }
                    }
                }
                cout << "\n���س�������...";
                cin.ignore();
                cin.get();
            }
            catch (const exception& e)
            {
                cout << "��������: " << e.what() << endl;
                cout << "\n���س�������...";
                cin.ignore();
                cin.get();
            }
        }
        // �Ƿ񱣴�
        cout << "�Ƿ񱣴浱ǰ����(y/n)��";
        char save;
        cin >> save;
        if (save == 'y' || save == 'Y')
        {
            cout << "�����뱣���ļ�����";
            string fname;
            cin >> fname;
            mat->saveToFile(fname);
            cout << "�ѱ��浽 " << fname << endl;
            cout << "\n���س�������...";
            cin.ignore();
            cin.get();
        }
        delete mat;
    }
}
int main()
{

    showMenu();
    return 0;
}
























