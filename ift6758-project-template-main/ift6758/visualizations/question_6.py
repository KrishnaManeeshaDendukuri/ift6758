import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from PIL import Image

#configure path
PLOT_PATH = "/ift6758/ift6758-blog-template-main/_includes/question_6_shotmap.html"

def prepare_shotmap_data(df, kernel_bw=1, n1=101, n2=101):
    
    """
    outputs a df ready to be fed to the funciton which builds the shot map visualisation
    
    parameters: 
        kernel_bw: bandwith for 2d gaussian kernel estimate
        n1 x n2: size of discrete grid for which we comute the probability densities estimated by the kernel
    """
    
    df_copy = df.copy()
    df_copy = df_copy[(~df_copy['x_coordinates'].isnull()) & (~df_copy['y_coordinates'].isnull())]
    
    # filter out playoffs : since we are looking at aggregate stats instead of scaling by time (as stated on piazza), 
    # it doesn't make sense to include playoffs when not all teams have made it to the same stage
    df_copy = df_copy[df_copy['game_id'].astype(str).str.slice(4,6) == '02']
    
    #if attacking team is on the right side, switch coordinates to the left so that we can have all coordinates on the same side
    df_copy = get_left_side_coords(df_copy)
    
    # filter out shots that happen on the other side of the red line (these shots are too rare to be interesting in our visualisation)
    df_copy = df_copy[df_copy['x_coordinates_left'] > 0]
   
    
    
    # set up for kernel density estimation
    output_df = pd.DataFrame()
    x = np.linspace(0, 100, n1)
    y = np.linspace(-42.5, 42.5, n2)
    xy = np.array(np.meshgrid(x, y)).reshape(2,-1)
    
    n_games_per_team = { # if we assume each game = 1 hour, shots per hour = shots in season / number of games
        20162017 : 82,
        20172018 : 82,
        20182019 : 82,
        20192020 : 70, # in reality, teams played 68-72 games since the season stopped abruptly due to covid
        20202021 : 56
    }

    
    # compute kernel for each season, compute kernel for each team x season, look at the difference between team and team x season
    
    for season in sorted(list(set(df_copy['season']))):
        
        df_season = df_copy[df_copy['season'] == season]
        xy_coords_season =  df_season[['x_coordinates_left','y_coordinates_left']].to_numpy().T
        kernel_season = gaussian_kde(xy_coords_season, bw_method = kernel_bw)
        prob_density_grid_season = kernel_season(xy) # this computes the density per sq feet per shot
        prob_density_grid_season = prob_density_grid_season * 100 * (len(df_season)/(len(set(df_season['attacking_team']))*n_games_per_team[season])) # scale to per hour per 100 sq ft
        
        for team in sorted(list(set(df_season['attacking_team']))):

            df_team_season = df_season[df_season['attacking_team'] == team]

            kernel_team_season = gaussian_kde(df_team_season[['x_coordinates_left','y_coordinates_left']].to_numpy().T, bw_method = kernel_bw)
            prob_density_grid_team_season = kernel_team_season(xy) # this computes the density per sq feet per shot
            prob_density_grid_team_season = prob_density_grid_team_season * 100 * (len(df_team_season)/n_games_per_team[season]) # scale to per hour per 100 sq ft
            team_season_differential = prob_density_grid_team_season - prob_density_grid_season

            output_df[f"{season} {team}"] = team_season_differential
                
    return output_df


def get_left_side_coords(df):
    """for shots from the right side, get corresponding left side coordinates"""
    
    df['x_coordinates_left'] = (df['attacking_team_side'] == 'left') * df['x_coordinates']  - (df['attacking_team_side'] == 'right') * df['x_coordinates']
    df['y_coordinates_left'] = (df['attacking_team_side'] == 'left') * df['y_coordinates']  - (df['attacking_team_side'] == 'right') * df['y_coordinates']
    
    return df


def plot_shotmap(shotmap_df, rink_image, n1=101, n2=101):
    """
    Produce shotmap visualisation
    
    parameters:
        shotmap_df: output of prepare_shotmap_data()
        rink_image: provided image of hockey rink
        n1, n2: same n1, n2 that were used in plot_shotmap()
    """
    
    rink_image = rink_image.crop((rink_image.size[0]/2, 0, rink_image.size[0], rink_image.size[1]))
    
    x = np.linspace(0, 100, n1)
    y = np.linspace(-42.5, 42.5, n2)
    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            z=np.array(shotmap_df[shotmap_df.columns[0]]).reshape(n2,n1),
            x=x,
            y=y,
            hoverongaps = False,
            opacity = 0.3,
            zmin=-0.55, zmax=0.55,
            colorbar=dict(title='Excess shots per hour per 100 sqft', titleside='right')
        )
    )

    updatemenus = [
        {
            'buttons': 
               [{'method': 'restyle', 'label': col,'args': [{'z': [np.array(shotmap_df[col]).reshape(n2,n1)]}]} for col in shotmap_df.columns],
            'direction': 'down',
            'showactive': True,

        }
    ]

    fig.layout = go.Layout(
        updatemenus=updatemenus,
        title="Shot Map for Team Relative to League-Wide Season Average"
    )

    img_width = rink_image.size[0]
    img_height = rink_image.size[1]
    scale_factor = 1/5.5


    fig.update_xaxes(
        visible=True,
        range=[0, 100 ]
    )

    fig.update_yaxes(
        visible=True,
        range=[-42.5, 42.5],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add images
    fig.add_layout_image(
            dict(
                source=rink_image,
                xref="x",
                yref="y",
                x=0,
                sizex=img_width * scale_factor,
                y=42.5,
                sizey=img_height * scale_factor,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
    )

    fig.update_layout(
        autosize=False,
        width=img_width*1.5,
        height=img_height*1.5)
    
    fig.write_html(PLOT_PATH)
    
    fig.show()
