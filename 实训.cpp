#include <iostream>
#include <vector>       //C++ 标准库中的 vector 容器的定义和相关功能引入到您的程序中。存储矩阵的数据： vector<vector<T>> data_
#include <fstream>      //对文件进行读写操作的类和函数。
#include <complex>      //可以进行复数类型相关运算
#include <stdexcept>    //引入标准异常类库
#include <cmath>        //引入数学函数和常量，开方sqrt，绝对值abs
#include <algorithm>    //引入算法库,swap和max函数
#include <sstream>      // 添加这个头文件支持字符串流，通过 istringstream 把一行字符串分割成多个数据，便于逐个读取和处理矩阵元素。
#include <limits>       // 添加这个头文件支持numeric_limits，用于计算该类型最大，最小，极限
#include <string>       // 确保包含string头文件
using namespace std;    //使用标准命名空间
const static constexpr double EPSILON = 1e-12;//设计一个很小的数字，用于判定浮点数是否为零
// 基类：定义矩阵公共接口（多态）
class MatrixBase
{
public:
    virtual ~MatrixBase() = 0;// 纯虚析构函数
    virtual void print() const = 0;//打印矩阵，纯虚函数
    virtual void saveToFile(const string& filename) const = 0;// // 保存到文件，纯虚函数
    virtual size_t getRows() const = 0;//获取行数
    virtual size_t getCols() const = 0;//获取列数
};
// 纯虚析构函数的定义
MatrixBase::~MatrixBase() {}
// 模板矩阵类
template<typename T>
class Matrix : public MatrixBase
{
private:
    size_t rows_;//大于等于零的整数，行 size_t 是一种无符号整数类型，用于表示非负整数
    size_t cols_;//列
    vector<vector<T>> data_;//外层 vector 是行，内层 vector 是列。

public:
    //进行输入输出重载，友元函数的声明
    template<typename T>//可以让友元函数支持不同于类模板参数的类型
    friend istream& operator>>(istream& ifs, Matrix<T>& mat);
    template<typename T>
    friend ostream& operator<<(ostream& ofs, Matrix<T>& mat);
    // 构造函数
    Matrix() : rows_(0), cols_(0), data_() {}  //无参默认构造函数：即为0行0列
    Matrix(size_t rows, size_t cols, const T& init = T())//指定元素的初始值，默认为T类型的默认值，引用传递可以避免创建对象的副本
        : rows_(rows), cols_(cols), data_(rows, vector<T>(cols, init)) {
    }
    //指定行列数和初始值的构造函数，创建一个rows行的动态矩阵，每个元素是cols列，初始值为init
    Matrix(const vector<vector<T>>& data)//常量引用避免不必要的拷贝
        : rows_(data.size()), cols_(data.empty() ? 0 : data[0].size()), data_(data) {
    }
    // 将矩阵的行数设置为传入的二维向量的外层大小，列数设为第一行的元素个数（假设所有行的长度相同），直接将传入的二维向量赋值给矩阵的内部数据成员 data_
 //vector 支持拷贝赋值 ，所以可以直接data_(data)
 // 拷贝构造函数
    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}
    // 基类虚函数实现
    void print() const override//代表这是纯虚函数的重写，const是函数签名的一部分。
    {    //auto表示自动类型推导，进行范围遍历循环
        for (const auto& row : data_) //遍历data_里面的行元素
        {
            for (const T& elem : row)//遍历行里面的列元素
            {
                if constexpr (is_same_v<T, complex<double>>)//条件编译语法，如果第一个不成立，不运行后面的
                {
                    double real = elem.real();//调用real() - 返回复数的实部
                    double imag = elem.imag();//调用imag() - 返回复数的虚
                    // 处理全零情况，实部虚部全为0的情况
                    if (abs(real) <= EPSILON && abs(imag) <= EPSILON)
                    {
                        cout << "0";
                    }
                    // 处理纯实数，虚部为0的情况
                    else if (abs(real) > EPSILON && abs(imag) <= EPSILON)
                    {
                        cout << real;
                    }
                    // 处理纯虚数，实部为0的情况
                    else if (abs(real) <= EPSILON && abs(imag) > EPSILON)
                    {
                        if (abs(abs(imag) - 1.0) <= EPSILON) //虚部绝对值为1，省略+-1
                        {
                            cout << (imag > 0 ? "" : "-") << "i";
                        }
                        else
                        {
                            cout << imag << "i";
                        }
                    }
                    // 处理既有实部又有虚部的复数
                    else
                    {
                        cout << real;
                        // 符号处理优化
                        cout << (imag > 0 ? "+" : "-");
                        double abs_imag = abs(imag);
                        if (abs(abs_imag - 1.0) <= EPSILON)
                        {
                            cout << "i";  // ：虚部绝对值为1时，省略+-1
                        }
                        else
                        {
                            cout << abs_imag << "i";
                        }
                    }
                }
                // 其他类型保持原有逻辑
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
                else//剩下例如整数类型直接输出
                {
                    cout << elem;
                }
                cout << "\t";//制表符用于对其输出
            }
            cout << endl;
        }
    }
    size_t getRows() const override
    {
        return rows_;
    }//重写虚函数，得到行数
    size_t getCols() const override
    {
        return cols_;
    }//重写虚函数，得到列数

    // Part 1：重载各种运算符重载
     // 重载赋值运算符
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
    //矩阵加法：算法：先判断矩阵行列是否相同，然后进行对应位置[i][j]的相加
    Matrix operator+(const Matrix& other) const
    {
        checkSameDimensions(other);//调用工具函数，检查维度是否相同，如果不相同，抛出异常
        Matrix result(rows_, cols_);//新创建一个矩阵，每个位置等于原来两个矩阵的元素相加
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result.data_[i][j] = data_[i][j] + other.data_[i][j];
            }
        }
        return result;
    }
    //矩阵减法：算法：先判断矩阵行列是否相同，然后进行对应位置[i][j]的相减，实现过程类似加法
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
    //矩阵乘法：特指两个矩阵相乘，算法：
    // 先判断A的列是否等于B的行，然后result的[i][j]等于A的i行对应元素乘B的j列的对应元素的积求和
    Matrix operator*(const Matrix& other) const
    {
        if (cols_ != other.rows_)
        {
            throw invalid_argument("维度不匹配，无法相乘");//进行判断
        }
        Matrix result(rows_, other.cols_);//调用构造函数，初始化为零
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
    // 添加矩阵与标量相乘的运算符,算法：对应位置的元素乘以标量
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
    // 添加友元函数（支持标量乘矩阵的交换律）2*A=A*2
    // 声明为友元主要是为了能够使用与类名相同的运算符重载语法
    friend Matrix operator*(const T& scalar, const Matrix& mat)
    {
        return mat * scalar; // 复用已有的矩阵乘标量运算符
    }
    //重载==运算符，判断两个矩阵是否完全相同，算法：对矩阵对应位置的每一个元素进行比较
    bool operator==(const Matrix& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)//先进行行列判断
        {
            return false;
        }
        for (size_t i = 0; i < rows_; ++i) //再进行对应位置的对应元素的大小判断
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                if (abs(data_[i][j] - other.data_[i][j]) > EPSILON) return false;
            }
        }
        return true;
    }
    // Part 2：矩阵运算
    // 矩阵转置
    // 定义：它将矩阵的行和列互换。对于一个 m×n 的矩阵 A，其转置矩阵 A^T 是一个 n×m 的矩阵。
    /* 算法：1.创建一个新矩阵，行数为原矩阵的列数，列数为原矩阵的行数
             2. 遍历原矩阵的每个元素
             3. 将原矩阵中位置(i, j) 的元素放到新矩阵的位置(j, i)*/
    Matrix<T> transpose() const
    {
        Matrix result(cols_, rows_);//创建新矩阵
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result.data_[j][i] = data_[i][j];//交换位置
            }
        }
        return result;
    }
    // 添加共轭转置方法
    Matrix<T> conjugateTranspose() const
    {
        Matrix<T> result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                if constexpr (is_same_v<T, complex<double>>)
                    //如果是复数类型，则取共轭
                {
                    result.data_[j][i] = conj(data_[i][j]);// 使用conj函数获取复数的共轭，
                    //这是complex头文件提供的函数
                }
                else
                {
                    result.data_[j][i] = data_[i][j];
                }
            }
        }
        return result;
    }
    //求矩阵的秩
    //数学定义：最大非零子式的阶数 ：矩阵中最大的非零行列式的阶数
    //算法；采用高斯消元法，把矩阵转化为阶梯形式，最终非零行的数量就是矩阵的秩。
    //逐列处理 ：从左到右遍历每一列
    //寻找主元 ：在当前列中找到第一个非零元素作为主元
    //行交换 ：将主元所在行与当前处理的行交换
    //行归一化 ：将主元所在行的主元归一化为1
    //消元 ：消去主元列下方的所有元素
    //计数 ：每找到一个主元，秩加1，（同时表示下一轮从下一行开始！！！！！不然会出错）
    int rank() const
    {
        Matrix mat = *this;// 创建矩阵副本，不改变原矩阵
        int rank = 0;
        for (size_t col = 0; col < cols_ && rank < rows_; ++col) //秩小于等于行数
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
    //将矩阵转换为行最简形式
    /*行最简形的特点:
    每个非零行的第一个非零元素（主元）为1
    每个主元所在列的其他元素都为0
    主元从左到右、从上到下排列
    所有全零行都在矩阵底部*/
    //算法：和求矩阵的秩类似，逐列处理，但要把当前列的其他元素都消为0，
    // 就相当于最后一步消元换一下，并且不用计数，最后返回矩阵的最简形式
    Matrix<T> reducedRowEchelonForm() const
    {
        Matrix mat = *this;
        int currentRow = 0;
        // 遍历每一列
        for (size_t col = 0; col < cols_ && currentRow < rows_; ++col)
        {
            // 寻找主元
            size_t pivot = findPivot(mat, col, currentRow);
            // 如果当前列没有非零元素，继续处理下一列
            if (pivot == rows_)
            {
                continue;
            }
            swapRows(mat, currentRow, pivot);
            normalizeRow(mat, currentRow, col);
            eliminateAllRows(mat, currentRow, col);
            // 移动到下一行
            currentRow++;
        }
        return mat;
    }
    // 计算方阵行列式的值=最后对角线的乘积*交换次数
    //算法：先判断是不是方阵，然后进行高斯消元法，要注意符号的判定
    T determinant() const
    {
        if (!isSquare())
        {
            throw invalid_argument("非方阵，无法计算行列式");
        }
        Matrix mat = *this;
        T det = 1;//用于记录行列式的值，初始化为1
        int swapCount = 0;//用于记录行交换的次数
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
    //求逆矩阵 ，逆矩阵 A^(-1) 满足 A·A^(-1) = A^(-1)·A = I
    //利用方法：A^(-1) = (1/det(A)) · adj(A)。det(A) 是行列式，adj(A) 是伴随矩阵
    //算法：先判断是不是方阵，然后计算行列式，如果行列式为零，表示矩阵不可逆，接着分别求
    //伴随矩阵和行列式，最后返回伴随矩阵除以行列式的结果
    //实数类型和复数类型的处理方式略有不同，复数矩阵是通过求模来判断是否为零
    auto inverse() const
    {
        if constexpr (is_same_v<T, complex<double>>) // 如果是复数类型
        {
            if (!isSquare())
            {
                throw invalid_argument("非方阵，不可逆");
            }
            complex<double> det = this->determinant();//计算行列式
            if (abs(det) < EPSILON)//若行列式为零，表示矩阵不可逆
            {
                throw runtime_error("矩阵不可逆");
            }
            Matrix<complex<double>> adj = this->adjugate();//计算伴随矩阵
            Matrix<complex<double>> result(this->getRows(), this->getCols());//创建结果矩阵
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
                throw invalid_argument("非方阵，不可逆");
            }
            T det = determinant();//计算行列式
            if (isZero(det))
            {
                throw runtime_error("矩阵不可逆");
            }
            Matrix<double> adj = this->adjugate();//计算伴随矩阵
            Matrix<double> result(this->getRows(), this->getCols());//创建结果矩阵
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
    // 求伴随矩阵
    //数学定义 ：伴随矩阵 adj(A) 的第 (j,i) 个元素是矩阵 A 的第 (i,j) 个元素的代数余子式
    //就相当于代数余子式矩阵的转置
    //算法：先判断是不是方阵，然后计算每个位置的代数余子式，最后转置得到伴随矩阵
    //代数余子式是去掉第 i 行和第 j 列后的子矩阵的行列式，乘以 (-1)^(i+j)
    Matrix<T> adjugate() const
    {
        if (!isSquare())
        {
            throw invalid_argument("非方阵");
        }
        Matrix<T> cof(rows_, cols_);//创建代数余子式矩阵
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                Matrix<T> minor = subMatrix(i, j);//获取去掉第 i 行和第 j 列后的子矩阵
                T sign = ((i + j) % 2 == 0) ? T(1) : T(-1);//计算符号
                cof.at(i, j) = sign * minor.determinant();//计算子矩阵的行列式，符号调整，赋值
            }
        }
        return cof.transpose(); // 伴随矩阵是代数余子式矩阵的转置
    }
    //把内容保存到文件里面
    void saveToFile(const string& filename) const override
    {
        ofstream ofs(filename);//打开文件进行写入
        if (!ofs) 
        {
            throw runtime_error("无法打开文件");
        }
        ofs << rows_ << " " << cols_ << endl;//先写入行和列
        for (const auto& row : data_)//遍历每一行，逐行写入每一行的元素
        {
            for (const T& elem : row)
            {
                ofs << elem << " ";
            }
            ofs << endl;
        }
    }
    // 工厂方法：从文件加载矩阵
    static Matrix<T> loadFromFile(const string& filename)
    {
        ifstream ifs(filename);//打开文件进行读取
        if (!ifs)
        {
            throw runtime_error("无法打开文件");
        }
        Matrix<T> mat;
        ifs >> mat;// 使用重载的输入运算符读取矩阵数据
        return mat;
    }
    // 在Matrix类的public部分添加以下声明
    vector<complex<double>> eigenvalues(int max_iterations = 1000) const;//计算矩阵的特征值的时候用到

    // 向量范数计算函数
    // p-范数计算，p=1为曼哈顿范数，p=2为欧几里得范数
    static double vectorNorm(const vector<T>& v, double p)
    {
        // 处理无穷范数情况 (p = inf)
        if (isinf(p)) //返回向量元素绝对值的最大值
        {
            double maxVal = 0.0;
            for (const auto& val : v)
            {
                double absVal = 0.0;
                if constexpr (is_same_v<T, complex<double>>)
                {
                    absVal = abs(val);  // 复数直接取模
                }
                else
                {
                    absVal = abs(static_cast<double>(val));  // 其他数值类型转换
                }
                maxVal = max(maxVal, absVal);
            }
            return maxVal;
        }
        // 计算p-范数，每个元素的p次方求和，最后在开根
        double sum = 0.0;
        for (const auto& val : v)
        {
            double absVal = 0.0;
            if constexpr (is_same_v<T, complex<double>>)
            {
                absVal = abs(val);  // 复数取模
            }
            else
            {
                absVal = abs(static_cast<double>(val));  // 数值类型转换
            }
            sum += pow(absVal, p);
        }
        return pow(sum, 1.0 / p);//利用pow函数，开p次方根
    }
    // 矩阵的Frobenius范数，矩阵所有元素绝对值的平方和再开根号
    double frobeniusNorm() const
    {
        double sum = 0.0;
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                if constexpr (is_same_v<T, complex<double>>)
                {
                    sum += norm(data_[i][j]); // 复数的模的平方
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

    // 矩阵的行和范数（无穷范数），每一行所有元素绝对值之和的最大值
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
                    rowSum += abs(data_[i][j]);//计算复数的模
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
    // 矩阵的列和范数，每一列所有元素绝对值之和的最大值
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
    //计算矩阵的谱范数，AA*的最大特征值的平方根，A*为共轭矩阵！！！不是转置矩阵
    double spectralNorm() const
    {
        if (rows_ == 0 || cols_ == 0)
        {
            return 0.0;
        }

        // 计算 A^H * A（共轭转置 * 原矩阵）
        Matrix<T> AH = this->conjugateTranspose();
        Matrix<T> AHA = AH * (*this);

        // 计算 A^H * A 的特征值
        auto eigenvals = AHA.eigenvalues();

        // 找出最大特征值的模
        double maxEigenvalue = 0.0;
        for (const auto& val : eigenvals)
        {
            double absVal = abs(val);
            if (absVal > maxEigenvalue)
            {
                maxEigenvalue = absVal;
            }
        }
        // 谱范数是最大特征值的平方根
        return sqrt(maxEigenvalue);
    }
    // 矩阵的条件数，就是矩阵的范数乘以逆矩阵的范数
    double conditionNumber(const string& normType) const
    {
        if (!isSquare())
        {
            throw invalid_argument("条件数只对方阵有定义");
        }
        try
        {
            T det = determinant();
            if (isZero(det))
            {
                return numeric_limits<double>::infinity();// 如果行列式为零，返回无穷大
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
                throw invalid_argument("未知范数类型: " + normType);
            }
            return normA * normInv;
        }
        catch (const exception&)//以常量引用方式捕获所有标准异常，例如矩阵不可逆等等
        {
            return numeric_limits<double>::infinity();//返回正无穷处理
        }
    }
    //使用at函数访问矩阵元素
    // 非const版本，返回可修改的引用
    T& at(size_t row, size_t col)
    {
        if (row >= rows_ || col >= cols_)
            throw out_of_range("矩阵索引越界");
        return data_[row][col];
    }
    // const版本，返回只读引用
    const T& at(size_t row, size_t col) const
    {
        if (row >= rows_ || col >= cols_)
            throw out_of_range("矩阵索引越界");
        return data_[row][col];
    }
    //把一个复数（complex<double>）格式化为字符串，用于友好地输出复数，
    //原来格式是（a，b）的形式,转换成a+bi
    static string formatComplex(const complex<double>& c)
    {
        stringstream ss;//创建一个字符串流对象
        const double EPSILON = 1e-12;
        double real = c.real();
        double imag = c.imag();
        //输出逻辑说明
        bool show_real = abs(real) > EPSILON;//先将实部和0进行比较
        bool show_imag = abs(imag) > EPSILON;//虚部和0比较
        if (show_real)
        {
            ss << real;
        }
        if (show_imag)//在虚部不为0的情况下
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
            if (abs(abs_imag - 1.0) > EPSILON)//判断虚部是不是1，在分情况
            {
                ss << abs_imag;
            }
            ss << "i";
        }
        if (!show_real && !show_imag)//实部虚部均为0
        {
            ss << "0";
        }
        return ss.str();//调用成员函数，获取字符串流中当前累积的内容，作为一个 string 类型返回
    }
    //求解线性方程组
    //返回值是一个二元组合类型，前者是解向量，后者是解的类型
    //解的类型：0表示无解，1表示唯一解，2表示无穷多解
    //算法：先创建增广矩阵，然后将其化为行最简形式，
    //接着计算系数矩阵和增广矩阵的秩，判断解的情况
    //接着根据系数矩阵的秩判断解的类型，
    //最后提取解向量，返回解向量和解的类型
    pair<vector<T>, int> solveLinearSystem(const vector<T>& b) const
    {
        if (b.size() != rows_)
        {
            throw invalid_argument("常数项向量的维度必须与矩阵行数相同");
        }
        // 创建增广矩阵 [A|b]
        Matrix<T> augmented(rows_, cols_ + 1);
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                augmented.at(i, j) = at(i, j);
            }
            augmented.at(i, cols_) = b[i];
        }
        // 将增广矩阵化为行最简形式
        Matrix<T> rref = augmented.reducedRowEchelonForm();
        // 计算系数矩阵的秩
        int rankA = rank();
        // 计算增广矩阵的秩
        int rankAb = rref.rank();
        // 检查是否有解，进行秩的比较
        if (rankA < rankAb)
        {
            // 如果系数矩阵的秩小于增广矩阵的秩，则无解
            return { vector<T>(), 0 };
        }
        // 在有解的前提下，判断解的情况
        int solutionType;
        if (rankA < cols_)
        {
            solutionType = 2; // 无穷多解
        }
        else
        {
            solutionType = 1; // 唯一解
        }
        // 紧接着求解向量
        vector<T> solution(cols_, T(0));// 初始化解向量为零
        // 找出主元，阶梯型矩阵中每一行第一个非零元素的位置
        vector<size_t> pivotCols;
        vector<size_t> pivotRows;
        size_t r = 0;// 当前行
        //从第一行开始，遍历每一列，寻找不为0的主元
        for (size_t j = 0; j < cols_ && r < rows_; ++j)
        {
            if (!isZero(rref.at(r, j)))
            {
                pivotCols.push_back(j);//记录主元所在的列到pivotCols
                pivotRows.push_back(r);//记录主元所在的行到pivotRows
                r++;
            }
        }
        // 对于每一个主元位置，提取对应的解，唯一解时，最后一列就是解！！！！
        //主元所在的每一行，最后一列就是对应未知数的解，因为前面已经化成了行最简形式.主元为1
        for (size_t i = 0; i < pivotCols.size(); ++i)
        {
            //主元所在列号                      主元所在行号，这一行最后一列的元素
            solution[pivotCols[i]] = rref.at(pivotRows[i], cols_);//列对应Xi=最后一个元素
        }
        return { solution, solutionType };
    }

    // 求解齐次线性方程组 AX=0 的基础解系
    //算法：先将矩阵化为行最简形式，找出主元列和自由列，
    //如果没有自由变量，则只有零解；否则为每个自由变量构造一个基础解向量
	//基础解向量就是有一个自由变量为1，其余自由变量为0，主元变量通过回代求解得到的向量

    vector<vector<T>> solveHomogeneousSystem() const
    {
        // 1. 化为行最简型
        Matrix<T> rref = this->reducedRowEchelonForm();

		// 2. 找主元列和自由变量列，自由变量就是那些没有主元的列
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
        // 剩下的列都是自由变量，找到所有的自由变量
        for (size_t j = 0; j < cols_; ++j)
        {
            if (find(pivotCols.begin(), pivotCols.end(), j) == pivotCols.end())
            //find(起始迭代器, 结束迭代器, 要查找的值) ，如果没找到，返回 pivotCols.end()      代表没找到
                freeCols.push_back(j);//添加到自由变量里面
        }

        // 3. 如果没有自由变量，只有零解
        if (freeCols.empty())
        {
            return {};
        }
        // 4. 为每个自由变量构造一个基础解向量
        vector<vector<T>> basisVectors;//基础解系
		for (size_t k = 0; k < freeCols.size(); ++k)// 对每个自由变量创建一个基础解向量
        {
			vector<T> sol(cols_, T(0));// 初始化解向量为零
            sol[freeCols[k]] = T(1); // 当前自由变量设为1
            // 回代求主元变量
			for (size_t i = 0; i < pivotCols.size(); ++i)//遍历主元列
            {
                size_t row = pivotRows[i];
                size_t col = pivotCols[i];
                T sum = T(0);
                // 注意：所有自由变量都要带入，即使是0
                for (size_t j = 0; j < freeCols.size(); ++j)
                {
                    sum += rref.at(row, freeCols[j]) * sol[freeCols[j]];
                    //计算主元所在行的其他自由变量的贡献
                }
				sol[col] = -sum;// 计算主元变量的值
            }
			basisVectors.push_back(sol);// 将基础解向量添加到结果中，依次添加每个向量
        }
        return basisVectors;
    }

	// 求解非齐次线性方程组 Ax=b 的通解，就相当于求特解和齐次方程组的基础解系
    pair<vector<T>, vector<vector<T>>> solveLinearSystemComplete(const vector<T>& b) const
    {
        pair<vector<T>, int> result = solveLinearSystem(b);
        vector<T> particularSolution = result.first;
        int solutionType = result.second;
        // 如果无解，直接返回
        if (solutionType == 0)
        {
            return { particularSolution, {} };
        }
        // 如果有唯一解，返回特解和空基础解系
        if (solutionType == 1)
        {
            return { particularSolution, {} };
        }
        // 如果有无穷多解，求齐次方程组的基础解系
        vector<vector<T>> basisVectors = solveHomogeneousSystem();
        return { particularSolution, basisVectors };
    }
	// 矩阵指数计算方法，这是用于求解线性微分方程组
    Matrix<T> matrixExponential(double t) const
    {
        if (!isSquare())
        {
            throw invalid_argument("矩阵指数只对方阵有定义");
        }
        // 使用泰勒级数计算矩阵指数 e^(At)
        Matrix<T> result = Matrix<T>::identity(rows_);
        Matrix<T> term = Matrix<T>::identity(rows_);
        double factorial = 1.0;
        // 计算前100项泰勒级数，这样才足够精确，不会有很大的误差
        for (int i = 1; i <= 100; ++i)
        {
            term = term * (*this) * (t / i);  // A^i * t^i / i!
            factorial *= i;//计算阶乘，但不太需要，可以删掉
            result = result + term;
            // 如果项变得非常小，可以提前终止
            if (term.frobeniusNorm() < EPSILON)
            {
                break;
            }
        }
        return result;
    }
	// 求解一阶线性齐次微分方程组 dx/dt = Ax, x(0) = x0，使用公式解是 x(t) = e^(At) * x0
    vector<T> solveLinearODE(const vector<T>& x0, double t) const
    {
        if (!isSquare()) 
        {
            throw invalid_argument("系数矩阵必须是方阵");
        }
        if (x0.size() != rows_)
        {
            throw invalid_argument("初始值向量维度必须与矩阵行数相同");
        }
        // 计算矩阵指数 e^(At)
        Matrix<T> expAt = matrixExponential(t);
        // 计算 x(t) = e^(At) * x0
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
	// 创建单位矩阵，在泰勒级数计算矩阵指数的时候用到。第一项就是单位矩阵
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
	// 添加类型转换方法，将原来类型的矩阵转换为其他类型的矩阵，兼容不同类型的矩阵运算，实数矩阵运算可能出现复数
    //可以把 Matrix<double> 转换为 Matrix<complex<double>>，或者反过来。
    template<typename U>
    Matrix<U> cast() const
    {
        Matrix<U> result(rows_, cols_);//
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                // 对于复数类型需要特殊处理
                if constexpr (is_same_v<T, complex<double>>)
                {
                    if constexpr (is_same_v<U, double>)
                    {
						result.at(i, j) = data_[i][j].real();// 只取实部
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
    // 判断矩阵是否为方阵
    bool isSquare() const { return rows_ == cols_; }
    // 检查维度是否相同（行列都要相同）
    void checkSameDimensions(const Matrix& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw invalid_argument("维度不匹配");
    }
    //判断数值是否为零
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
	//计算代数余子式的时候使用，求子矩阵
    Matrix<T> subMatrix(size_t excludeRow, size_t excludeCol) const
    {
        if (rows_ <= 1 || cols_ <= 1)
        {
            throw std::invalid_argument("无法对1x1或更小的矩阵取子矩阵");
        }
        if (excludeRow >= rows_ || excludeCol >= cols_)
        {
            throw std::out_of_range("subMatrix参数越界");
        }
        Matrix<T> result(rows_ - 1, cols_ - 1);
        for (size_t i = 0, r = 0; i < rows_; ++i)
        {
            if (i == excludeRow)
            {
				continue;// 跳过要排除的行
            }
            for (size_t j = 0, c = 0; j < cols_; ++j)
            {
                if (j == excludeCol)
                {
                    continue;//跳过要排除的列
                }
                result.data_[r][c++] = data_[i][j];
            }
            r++;
        }
        return result;
    }
    //static作用：这些工具函数函数不需要访问特定对象的非静态成员变量，它们只操作传入的矩阵参数。
    //寻找主元，工具函数，在行列式化简和求矩阵的秩的时候用于寻找当前列规定范围内第一个非零元素的值
    //就是列一定，寻找第一个不为零的元素的行
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
        if (maxAbs < EPSILON) return mat.rows_; // 全零
        return pivot;
    }
    //交换行，也是一个工具函数，用于把当前行和主元所在行进行交换
    static void swapRows(Matrix& mat, size_t a, size_t b)
    {
        swap(mat.data_[a], mat.data_[b]);//调用algorithm库里面的swap函数用于交换
    }
    //工具函数，将主元（pivot）位置的值变为1，便于后续计算和化简为行最简型矩阵
    //算法：找到所在行，然后同除
    static void normalizeRow(Matrix& mat, size_t row, size_t col)
    {
        T pivot = mat.data_[row][col];//找到主元所在的行和列
        if (mat.isZero(pivot))//如果主元为零，不需要化为1
        {
            return;
        }
        for (size_t j = col; j < mat.cols_; ++j)
        {
            mat.data_[row][j] /= pivot;
        }
    }
    //工具函数，消去主元下方的行，专门用于求矩阵的秩
    static void eliminateRows(Matrix& mat, size_t row, size_t col)
    {
        for (size_t i = row + 1; i < mat.rows_; ++i)
        {
            T factor = mat.data_[i][col];//因为主元为1，所以直接乘以当前元素的负数，然后对应元素相加
            for (size_t j = col; j < mat.cols_; ++j)
            {
                mat.data_[i][j] -= factor * mat.data_[row][j];
            }
        }
    }
    //工具函数，会将主元列上除了主元行以外的所有非零元素都消为0，专门用于求行最简型矩阵
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
    //工具函数，在不把主元化成1的情况下面进行消元的操作，在计算行列式的值的时候使用
    //并且传入的参数也不一样
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
// 复数矩阵专用行列式实现（模板特化）
// 因为复数矩阵求行列式的值的时候和实数矩阵有一点区别，
// 复数是要用模比较，工具函数不能直接使用使用这里特化
template<>//模板特化声明
complex<double> Matrix<complex<double>>::determinant() const
{
    if (!isSquare()) 
    {
        throw invalid_argument("非方阵，无法计算行列式");
    }
    Matrix<complex<double>> mat = *this;
	complex<double> det = 1.0;// 初始化行列式为1
	int swapCount = 0;// 记录行交换次数
	size_t n = mat.rows_;// 获取矩阵的行数
    for (size_t col = 0; col < n; ++col) 
    {
        // 选主元（模最大）
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
        if (maxAbs < EPSILON) return complex<double>(0.0, 0.0); // 行列式为0
        if (pivot != col) //行交换
        {
            swap(mat.data_[col], mat.data_[pivot]);
            swapCount++;
        }
        det *= mat.data_[col][col];
        // 消元
        for (size_t i = col + 1; i < n; ++i) 
        {
            complex<double> factor = mat.data_[i][col] / mat.data_[col][col];
            for (size_t j = col; j < n; ++j)
            {
                mat.data_[i][j] -= factor * mat.data_[col][j];
            }
        }
    }
	if (swapCount % 2)// 如果交换行的次数是奇数，则行列式取负
    {
        det = -det;
    }
    // 若实部或虚部极小，置零
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
// 实数矩阵特征值（支持复特征值，2x2块直接解特征多项式）
//算法：使用QR算法迭代求解特征值，直到矩阵近似上三角形式
// 迭代次数可以通过参数 maxIterations 控制，默认100次
// 复数矩阵特征值计算使用了QR分解和迭代方法，
// 通过Gram-Schmidt正交化过程构造正交矩阵Q和上三角矩阵R，
// 然后迭代更新矩阵A，直到收敛为止
// 最后提取特征值，支持2x2块复特征值

template<>
vector<complex<double>> Matrix<double>::eigenvalues(int maxIterations) const //用复数矩阵存放结果
{
    // 检查是否为方阵
    size_t n = this->getRows();
    if (n != this->getCols()) throw invalid_argument("特征值只能计算方阵。");
    // 拷贝数据
    vector<vector<double>> A(n, vector<double>(n));//创建A，进行拷贝
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            A[i][j] = this->at(i, j);
    double tolerance = 1e-10;// 收敛容忍度
    for (int iter = 0; iter < maxIterations; ++iter)//460行的时候函数声明给了默认值
    {
        // Gram-Schmidt QR分解
		vector<vector<double>> Q(n, vector<double>(n, 0));// 正交矩阵Q，存储的是【e1,e2,e3....】
		vector<vector<double>> R(n, vector<double>(n, 0));// 上三角矩阵R
        //首先先把矩阵写成列向量的形式
        for (size_t j = 0; j < n; ++j)//遍历每一列
        {   
			vector<double> v(n);// 用于存储当前列向量
            for (size_t i = 0; i < n; ++i)
            {
				v[i] = A[i][j];// 取当前列向量,进行Gram-Schmidt正交化，第j列
            }
            for (size_t k = 0; k < j; ++k) 
            {
				double proj = 0, normQ = 0;// 计算投影和Q的范数的平方
                for (size_t i = 0; i < n; ++i)
                {
					proj += v[i] * Q[i][k];// 计算投影，相当于先是e1*e2
					normQ += Q[i][k] * Q[i][k];// 计算Q的范数的平方，就是e2的模的平方
                }
                if (normQ < tolerance)
                {
					normQ = tolerance;// 防止除以零
                }
				proj = proj / normQ;// 计算投影系数，e1的转置乘以ak
                for (size_t i = 0; i < n; ++i)
					v[i] -= proj * Q[i][k];// 从当前列向量中减去投影部分，得到正交化后的向量
				    R[k][j] = proj;// 存储投影系数
            }
            double norm_val = 0.0;
            //把e1单位化，之后也一样
            for (size_t i = 0; i < n; ++i) norm_val += v[i] * v[i];//计算模长的平方
            norm_val = sqrt(norm_val);// 计算当前列向量的模长
            if (norm_val < tolerance)
            {
                v[j] = 1;
                norm_val = 1.0;
            }
            for (size_t i = 0; i < n; ++i)
                Q[i][j] = v[i] / norm_val;//Q用来存储的就相当于是【e1.e2,e3....】
                R[j][j] = norm_val;
        }
        // A = R * Q
        vector<vector<double>> newA(n, vector<double>(n, 0));// 新的矩阵A
        // 计算新矩阵A = R * Q
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < n; ++k)
                    newA[i][j] += R[i][k] * Q[k][j];
        // 检查是否近似上三角
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
    // 提取特征值（支持2x2块复特征值）
    vector<complex<double>> eigenvals;
    size_t i = 0;
    while (i < n) 
    {
        if (i + 1 < n && fabs(A[i + 1][i]) > 1e-6) {
            // 2x2块，解 λ^2 - tr*λ + det = 0
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
// 在类外实现输出运算符重载
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
    is >> mat.rows_ >> mat.cols_;//先输入行和列
    //vector里面的函数
    mat.data_.resize(mat.rows_, vector<T>(mat.cols_));//确保矩阵的存储空间和你指定的行数、列数一致
    // 跳过输入流中直到遇到换行符为止的所有字符，常用于清空缓冲区或丢弃当前行剩余内容。
    is.ignore(numeric_limits<streamsize>::max(), '\n');
    for (size_t i = 0; i < mat.rows_; ++i)
    {
        string line;
        getline(is, line);
        // 如果读取到空行，重试
        if (line.empty() && i < mat.rows_)
        {
            i--;
            continue;
        }
        istringstream iss(line);//转换成流，方便输出 iss >> 变量
        for (size_t j = 0; j < mat.cols_; ++j)
        {
            T value;
            if (!(iss >> value))
            {
                //这里使用to_string，将数值类型转换为字符串。
                throw runtime_error("无效的矩阵元素格式，行 " + to_string(i + 1) + " 列 " + to_string(j + 1));
            }
            mat.at(i, j) = value;  // 使用at()方法
        }
    }
    return is;
}
// 特化复数矩阵的输入输出运算符重载
template<>
istream& operator>>(istream& is, Matrix<complex<double>>& mat)
{
    // 读取行列数
    if (!(is >> mat.rows_ >> mat.cols_))
    {
        throw runtime_error("无效的行列数格式");
    }
    mat.data_.resize(mat.rows_, vector<complex<double>>(mat.cols_));

    is.ignore(numeric_limits<streamsize>::max(), '\n');
    for (size_t i = 0; i < mat.rows_; ++i) {
        string line;
        getline(is, line);
        // 如果读取到空行，重试
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
                throw runtime_error("无效的矩阵元素格式，行 " + to_string(i + 1) + " 列 " + to_string(j + 1));
            }
            // 处理(real,imag)格式的复数
            if (input.front() == '(' && input.back() == ')')
            {
                // 去除括号
                input = input.substr(1, input.size() - 2);//起始位置，要截取的长度
                // 查找逗号位置
                size_t comma_pos = input.find(',');
                if (comma_pos != string::npos)//表明存在
                {
                    double real = stod(input.substr(0, comma_pos));//stdod将字符串转换为double
                    double imag = stod(input.substr(comma_pos + 1));
                    mat.at(i, j) = complex<double>(real, imag);  // 使用at()方法
                    continue;
                }
            }
            // 如果不是(real,imag)格式，尝试使用a+bi格式解析
            complex<double> value(0, 0);
            try {
                // 去除输入字符串中可能存在的空格，erase(first, last); // 删除[first, last)区间的所有字符
                input.erase(remove_if(input.begin(), input.end(), ::isspace), input.end());
                // 检查是否为纯虚数形式 (i, -i, 3i, -3i)
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
                    // 处理其他形式的纯虚数 (如 3i, -3i)
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
                // 处理带有加号的复数 (a+bi)
                if (input.find('+') != string::npos && input.find('i') != string::npos)
                {
                    size_t plus_pos = input.find('+');
                    double real_part = stod(input.substr(0, plus_pos));//提取实部
                    string imag_str = input.substr(plus_pos + 1);//提取虚部
                    double imag_part = 1.0;// 默认虚部为1
                    if (imag_str != "i")
                    {
                        imag_part = stod(imag_str.substr(0, imag_str.size() - 1));
                    }
                    value = complex<double>(real_part, imag_part);
                }
                // 处理带有减号的复数 (a-bi)
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
                // 处理纯虚数形式 (i, -i, 3i, -3i)
                else if (input.back() == 'i')
                {
                    // 处理其他形式的纯虚数 (如 3i, -3i)
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
                // 处理纯实数
                else
                {
                    value = complex<double>(stod(input), 0);
                }
            }
            catch (const exception& e)
            {
                throw runtime_error("无效的复数格式: " + input + " (" + e.what() + ")");
            }

            mat.at(i, j) = value;  // 使用at()方法
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
    // 处理实部显示
    bool show_real = abs(real) > EPSILON;
    // 处理虚部显示
    bool show_imag = abs(imag) > EPSILON;
    if (show_real) os << real;
    if (show_imag)
    {
        // 处理符号
        if (show_real)
        {
            os << (imag > 0 ? "+" : "-");
        }
        else if (imag < 0) {
            os << "-";
        }
        // 处理虚部数值
        double abs_imag = abs(imag);
        if (abs(abs_imag - 1.0) > EPSILON)
        {
            os << abs_imag;
        }
        os << "i";
    }
    // 处理全零情况
    if (!show_real && !show_imag) os << "0";
    return os;
}
// 在类外添加以下实现
// 复数矩阵QR分解+特征值（Gram-Schmidt版）
template<>
vector<complex<double>> Matrix<complex<double>>::eigenvalues(int maxIterations) const
{
    size_t n = this->getRows();
    if (n != this->getCols()) throw invalid_argument("特征值只能计算方阵。");

    // 拷贝数据
    vector<vector<complex<double>>> A(n, vector<complex<double>>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            A[i][j] = this->at(i, j);

    double tolerance = 1e-10;
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // Gram-Schmidt QR分解
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
        // 检查收敛
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
    // 提取对角线为特征值
    vector<complex<double>> eigenvals(n);
    for (size_t i = 0; i < n; ++i)
        eigenvals[i] = A[i][i];
    return eigenvals;
}
void showMenu()
{
    while (true)
    {
        system("cls"); // 清屏
        cout << "矩阵操作程序\n";
        cout << "1. 创建新矩阵并输入数据\n";
        cout << "2. 从文件加载矩阵\n";
        cout << "3. 退出程序\n";
        cout << "请选择操作（1-3）: ";
        int inputType;
        cin >> inputType;
        if (inputType == 3) break;
        MatrixBase* mat = nullptr;// 矩阵基类指针，用于存储不同类型的矩阵
        bool isComplex = false;//
        if (inputType == 1)
        {
            system("cls");
            cout << "请选择矩阵类型：\n";
            cout << "1. 实数矩阵\n";
            cout << "2. 复数矩阵\n";
            cout << "请选择（1-2）：";
            int type;
            cin >> type;
            system("cls");
            if (type == 1)
            {
                auto* m = new Matrix<double>();
                cout << "请输入矩阵（行 列，接着每行输入元素）：\n";
                cin >> *m;
                mat = m;
            }
            else
            {
                auto* m = new Matrix<complex<double>>();
                cout << "请输入复数矩阵（行 列，接着每行输入元素，复数如1+2i）：\n";
                cin >> *m;
                mat = m;
                isComplex = true;
            }
        }
        else if (inputType == 2)
        {
            system("cls");
            cout << "请输入文件名：";
            string filename;
            cin >> filename;
            cout << "请选择矩阵类型：\n";
            cout << "1. 实数矩阵\n";
            cout << "2. 复数矩阵\n";
            cout << "请选择（1-2）：";
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

        // 运算菜单
        while (true)
        {
            system("cls");
            cout << "\n矩阵运算选项：\n";
            cout << "1. 矩阵转置\n";
            cout << "2. 矩阵加法\n";
            cout << "3. 矩阵减法\n";
            cout << "4. 矩阵乘法\n";
            cout << "5. 求矩阵的秩\n";
            cout << "6. 化简到行最简型矩阵\n";
            cout << "7. 计算向量范数\n";
            cout << "8. 计算矩阵范数\n";
            cout << "9. 计算矩阵条件数\n";
            cout << "10. 计算方阵特征值\n";
            cout << "11. 计算方阵行列式\n";
            cout << "12. 计算矩阵的逆\n";
            cout << "13. 计算伴随矩阵\n";
            cout << "14. 求解一阶线性齐次微分方程组 dx/dt=Ax\n";
            cout << "15. 求解线性方程组 AX=b\n";
            cout << "16. 返回主菜单\n";
            cout << "请选择操作（1-16）：";
            int op;
            cin >> op;
            system("cls");
            if (op == 16) break;
            try
            {
                // 先显示原矩阵
                cout << "原矩阵为：\n";
                if (isComplex)
                    static_cast<Matrix<complex<double>>*>(mat)->print();
                else
                    static_cast<Matrix<double>*>(mat)->print();
                cout << endl;

                if (op == 1)
                {
                    cout << "转置结果：\n";
                    if (isComplex)
                        static_cast<Matrix<complex<double>>*>(mat)->transpose().print();
                    else
                        static_cast<Matrix<double>*>(mat)->transpose().print();
                }
                else if (op == 2 || op == 3 || op == 4)
                {
                    cout << "请输入另一个矩阵（行 列，接着每行输入元素）：\n";
                    if (isComplex)
                    {
                        Matrix<complex<double>> m2;
                        cin >> m2;
                        cout << "\n原矩阵为：\n";
                        static_cast<Matrix<complex<double>>*>(mat)->print();
                        cout << "\n另一个矩阵为：\n";
                        m2.print();
                        cout << "\n结果为：\n";
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
                        cout << "\n原矩阵为：\n";
                        static_cast<Matrix<double>*>(mat)->print();
                        cout << "\n另一个矩阵为：\n";
                        m2.print();
                        cout << "\n结果为：\n";
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
                    cout << "矩阵的秩为：";
                    if (isComplex)
                        cout << static_cast<Matrix<complex<double>>*>(mat)->rank() << endl;
                    else
                        cout << static_cast<Matrix<double>*>(mat)->rank() << endl;
                }
                else if (op == 6)
                {
                    cout << "行最简型矩阵为：\n";
                    if (isComplex)
                        static_cast<Matrix<complex<double>>*>(mat)->reducedRowEchelonForm().print();
                    else
                        static_cast<Matrix<double>*>(mat)->reducedRowEchelonForm().print();
                }
                else if (op == 7)
                {
                    cout << "请输入向量长度和元素：\n";
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
                        cout << "请输入p值（如1,2）：";
                        double p;
                        cin >> p;
                        cout << "范数为：" << Matrix<complex<double>>::vectorNorm(v, p) << endl;
                    }
                    else
                    {
                        vector<double> v(n);
                        for (size_t i = 0; i < n; ++i) cin >> v[i];
                        cout << "请输入p值（如1,2）：";
                        double p;
                        cin >> p;
                        cout << "范数为：" << Matrix<double>::vectorNorm(v, p) << endl;
                    }
                }
                else if (op == 8)
                {
                    if (isComplex)
                    {
                        auto* m = static_cast<Matrix<complex<double>>*>(mat);
                        cout << "Frobenius范数: " << m->frobeniusNorm() << endl;
                        cout << "行和范数: " << m->rowSumNorm() << endl;
                        cout << "列和范数: " << m->columnSumNorm() << endl;
                        cout << "谱范数: " << m->spectralNorm() << endl;
                    }
                    else
                    {
                        auto* m = static_cast<Matrix<double>*>(mat);
                        cout << "Frobenius范数: " << m->frobeniusNorm() << endl;
                        cout << "行和范数: " << m->rowSumNorm() << endl;
                        cout << "列和范数: " << m->columnSumNorm() << endl;
                        cout << "谱范数: " << m->spectralNorm() << endl;
                    }
                }
                else if (op == 9)
                {
                    cout << "请选择范数类型（frobenius/row/column/spectral）：";
                    string normType;
                    cin >> normType;
                    if (isComplex)
                        cout << "条件数: " << static_cast<Matrix<complex<double>>*>(mat)->conditionNumber(normType) << endl;
                    else
                        cout << "条件数: " << static_cast<Matrix<double>*>(mat)->conditionNumber(normType) << endl;
                }
                else if (op == 10)
                {
                    cout << "特征值为：\n";
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
                        cout << "行列式: " << Matrix<complex<double>>::formatComplex(static_cast<Matrix<complex<double>>*>(mat)->determinant()) << endl;
                    else
                        cout << "行列式: " << static_cast<Matrix<double>*>(mat)->determinant() << endl;
                }
                else if (op == 12)
                {
                    cout << "逆矩阵为：\n";
                    if (isComplex)
                    {
                        static_cast<Matrix<complex<double>>*>(mat)->inverse().print();
                        auto& A = *static_cast<Matrix<complex<double>>*>(mat);
                        auto inv = A.inverse();
                        auto prod = A * inv;
                        prod.print(); // 检查是否为单位矩阵
                    }
                    else
                        static_cast<Matrix<double>*>(mat)->inverse().print();
                }
                else if (op == 13)
                {
                    cout << "伴随矩阵为：\n";
                    if (isComplex)
                        static_cast<Matrix<complex<double>>*>(mat)->adjugate().print();
                    else
                        static_cast<Matrix<double>*>(mat)->adjugate().print();
                }
                else if (op == 14)
                {
                    cout << "请输入初始向量x0（每个元素用空格分隔）：\n";
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
                        cout << "请输入t的值: ";
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
                        cout << "请输入t的值: ";
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
                    cout << "请输入常数向量b（每个元素用空格分隔）：\n";
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
                            cout << "线性方程组无解。" << endl;
                        }
                        else
                        {
                            cout << "特解: ";
                            for (const auto& v : particular) cout << Matrix<complex<double>>::formatComplex(v) << "\t";
                            cout << endl;
                            if (!basis.empty())
                            {
                                cout << "基础解系:" << endl;
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
                            cout << "线性方程组无解。" << endl;
                        }
                        else
                        {
                            cout << "特解: ";
                            for (const auto& v : particular) cout << v << "\t";
                            cout << endl;
                            if (!basis.empty())
                            {
                                cout << "基础解系:" << endl;
                                for (const auto& vec : basis)
                                {
                                    for (const auto& v : vec) cout << v << "\t";
                                    cout << endl;
                                }
                            }
                        }
                    }
                }
                cout << "\n按回车键继续...";
                cin.ignore();
                cin.get();
            }
            catch (const exception& e)
            {
                cout << "发生错误: " << e.what() << endl;
                cout << "\n按回车键继续...";
                cin.ignore();
                cin.get();
            }
        }
        // 是否保存
        cout << "是否保存当前矩阵？(y/n)：";
        char save;
        cin >> save;
        if (save == 'y' || save == 'Y')
        {
            cout << "请输入保存文件名：";
            string fname;
            cin >> fname;
            mat->saveToFile(fname);
            cout << "已保存到 " << fname << endl;
            cout << "\n按回车键继续...";
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
























