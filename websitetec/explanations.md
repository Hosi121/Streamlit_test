### MNISTデータを使った多層パーセプトロンの伝播式と誤差逆伝播

$$
h^{(l)} = f\left(\sum_{i=1}^{n} w_{i}^{(l-1)} x_{i} + b^{(l-1)}\right)
$$

ここで \( h^{(l)} \) は \( l \) 層目のノードの活性化関数の出力、\( f \) は活性化関数、
\( w_{i}^{(l-1)} \) は \( l-1 \) 層目と \( l \) 層目の間の重み、\( x_{i} \) は \( l-1 \) 層目のノードの出力、
\( b^{(l-1)} \) は \( l \) 層目のバイアス項です。

誤差逆伝播では、出力層での誤差を用いて以下のように重みを更新します：

$$
\Delta w_{ij} = -\eta \frac{\partial \mathcal{L}}{\partial w_{ij}}
$$

ここで \( \Delta w_{ij} \) は重み \( w_{ij} \) の更新量、\( \eta \) は学習率、
\( \mathcal{L} \) は損失関数です。偏微分 \( \frac{\partial \mathcal{L}}{\partial w_{ij}} \) は、
損失関数の重み \( w_{ij} \) に対する勾配を表しています。

同様に、バイアスの更新は以下のように行われます：

$$
\Delta b_{i} = -\eta \frac{\partial \mathcal{L}}{\partial b_{i}}
$$

ここで \( \Delta b_{i} \) はバイアス \( b_{i} \) の更新量です。

個々の重みの勾配は、以下の連鎖律を用いて計算されます：

$$
\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial h_{j}} \cdot f'(h_{j}) \cdot x_{i}
$$

ここで \( f'(h_{j}) \) は活性化関数の導関数です。
