import jax
import jax.numpy as jnp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def draw_lineplot(x:jnp.array, 
                  y:jnp.array, 
                  x_label:str, 
                  y_label:str,
                  save_path):

    data = {x_label: x, y_label:y}
    df = pd.DataFrame(data=data)
    sns.lineplot(x=x_label, y=y_label, data=df)
    plt.savefig(save_path)