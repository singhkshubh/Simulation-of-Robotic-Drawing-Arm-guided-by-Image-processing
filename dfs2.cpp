#include <bits/stdc++.h>
#include <vector>
using namespace std;
const int p = 100;
float a[p][p];

int vis[p][p] = {0};

vector<float> b;
vector<float> c;
vector<float> d;
void dfs(int x, int y)
{
    vis[x][y] = 1;
    b.push_back(x);
    b.push_back(y);
    b.push_back(1);
    if (a[x - 1][y] == 1.0 && !vis[x - 1][y])
    {
        dfs(x - 1, y);
    }
    if (a[x][y - 1] == 1.0 && !vis[x][y - 1])
    {
        dfs(x, y - 1);
    }
    if (a[x + 1][y] == 1.0 && !vis[x + 1][y])
    {
        dfs(x + 1, y);
    }
    if (a[x][y + 1] == 1.0 && !vis[x][y + 1])
    {
        dfs(x, y + 1);
    }
    if (a[x + 1][y + 1] == 1.0 && !vis[x + 1][y + 1])
    {
        dfs(x + 1, y + 1);
    }
    if (a[x - 1][y + 1] == 1.0 && !vis[x - 1][y + 1])
    {
        dfs(x - 1, y + 1);
    }
    if (a[x + 1][y - 1] == 1.0 && !vis[x + 1][y - 1])
    {
        dfs(x + 1, y - 1);
    }
    if (a[x - 1][y - 1] == 1.0 && !vis[x - 1][y - 1])
    {
        dfs(x - 1, y - 1);
    }
    else
    {
        b.push_back(x);
        b.push_back(y);
        b.push_back(0);
    }
}

int main()
{

    ifstream f("dogr.txt");

    for (int i = 0; i < p; i++)
    {
        for (int j = 0; j < p; j++)
        {
            f >> a[i][j];
        }
    }
    b.push_back(0);
    b.push_back(0);
    b.push_back(0);
    for (int i = 0; i < p; i++)
    {
        for (int j = 0; j < p; j++)
        {
            if (a[i][j] == 1.0 && vis[i][j] == 0)
            {
                dfs(i, j);
            }
        }
    }

    for (int i = 1; i < ((b.size()) / 3); i++)
    {

        if (b.at(3 * i + 2) == 0 && b.at(3 * (i - 1) + 2) == 0)
        {

            continue;
        }
        else
        {

            c.push_back(b.at(3 * i));
            c.push_back(b.at(3 * i + 1));
            c.push_back(b.at(3 * i + 2));
        }
    }
    d.push_back(0);
    d.push_back(0);
    d.push_back(0);
    d.push_back(c.at(0));
    d.push_back(c.at(1));
    d.push_back(c.at(2));
    d.push_back(c.at(3));
    d.push_back(c.at(4));
    d.push_back(c.at(5));

    for (int i = 2; i < ((c.size()) / 3); i++)
    {
        if (c.at(3 * i + 2) == 1 && c.at(3 * (i - 2) + 2) == 1 && c.at(3 * (i - 1) + 2) == 1 && (((c.at(3 * (i - 2))) * (c.at(3 * i + 1)) + (c.at(3 * (i - 1))) * (c.at(3 * (i - 2) + 1)) + (c.at(3 * i)) * (c.at(3 * (i - 1) + 1))) - ((c.at(3 * (i - 2))) * (c.at(3 * (i - 1) + 1)) + (c.at(3 * (i - 1))) * (c.at(3 * i + 1)) + (c.at(3 * i)) * (c.at(3 * (i - 2) + 1)))) == 0)
        {
            continue;
        }
        else
        {

            d.push_back(c.at(3 * (i - 1)));
            d.push_back(c.at(3 * (i - 1) + 1));
            d.push_back(c.at(3 * (i - 1) + 2));
            d.push_back(c.at(3 * i));
            d.push_back(c.at(3 * i + 1));
            d.push_back(c.at(3 * i + 2));
        }
    }

    ofstream g("dogg.txt");

    g << "x y pen";
    g << endl;
    for (int i = 0; i < ((d.size()) / 3); i++)
    {
        g << d.at(3 * i);
        g << " ";
        g << d.at(3 * i + 1);
        g << " ";
        g << d.at(3 * i + 2);
        g << endl;
    }
    return 0;
}