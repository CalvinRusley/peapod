#!/usr/bin/env python
import numpy as np
import pandas as pd
import bokeh.io
from bokeh.models import HoverTool
import holoviews as hv
from bokeh.plotting import show
bokeh.io.output_notebook()
hv.extension('bokeh')
from holoviews import opts


def imshow(matrix, plot_height=400, interpixel_distance=1.0):
    """ Display a numpy array as an image in a Bokeh figure. x dimension corresponds to array dim0,
        y dimensions to array dim1.

        Parameters:
            matrix (matrix object): 2-dimensional similarity or scoring matrix
            plot_height (integer): height of the plot in pixels (default 400) (width is scaled accordingly)
            interpixel_distance (float): interpixel distance in units of pixels (default 1.0)

        Returns:
            holoviews plot
    """
    #modified from Justin Bois (https://justinbois.github.io/bootcamp/2017/lessons/l42_bokeh_holoviews.html)
    # Get shape, dimensions
    n, m = matrix.array.shape
    dw = n * interpixel_distance
    dh = m * interpixel_distance
    
    # Set up figure with appropriate dimensions
    bounds = (0,0,dw,dh) #left bottom right top

    # Set color mapper
    ncolors = 256
    cmin = np.min(matrix.array)
    cmax = np.max(matrix.array)
    cmid = 0.0
    c_diverge_point_normalized = (cmid - cmin) / (cmax - cmin)
    palette_cutoff = round(c_diverge_point_normalized * ncolors)
    colors = bokeh.palettes.diverging_palette(
        bokeh.palettes.Blues[ncolors][(ncolors-palette_cutoff):],
        bokeh.palettes.Reds[ncolors],
        n=ncolors,
        midpoint=c_diverge_point_normalized,
    )
    color_mapper = colors

    # Display the image
    #flip the tensor so it is the right way round:
    im_flipped = np.flip(np.transpose(matrix.array), [0])
    #im_flipped = np.transpose(matrix.array)
    im = hv.Image(
        im_flipped, bounds=bounds
    ).opts(
        cmap=color_mapper,
        height=plot_height,
        aspect='equal',
        invert_yaxis=True,
        xaxis='top',
        xlabel='-'.join(matrix.deflines0),
        ylabel='-'.join(matrix.deflines1),
    )
    
    return im


def plot_positions(positions,size=3):
    """ Plots the coordinates of a positions object. x dimension corresponds to array dim0,
        y dimensions to array dim1. For overlaying on the result of imshow().

        Parameters:
            positions (positions object): positions to plot
            size (integer): marker size (default 3)

        Returns:
            holoviews plot
    """
    alignment_pixels = positions.coordinates + 0.5
    alignment_df = pd.DataFrame({'Column0': alignment_pixels[:, 0].tolist(), 'Column1': alignment_pixels[:, 1].tolist()})
    points = hv.Points(
        data=alignment_df
    ).opts(
        marker='square',
        xlabel='-'.join(positions.deflines0),
        ylabel='-'.join(positions.deflines1),
        size=size)
    return points


def summarize(matrix, positions_list, color_list=['#ffff00','#03fc03','#aa00ff'], plot_height=400, size=3):
    """ Plots a matrix overlain by multiple positions objects. x dimension corresponds to array dim0,
        y dimensions to array dim1.

        Parameters:
            matrix (matrix object): 2-dimensional similarity or scoring matrix
            positions_list (list of positions objects): list of positions to plot (up to 3, after which you need to provide your own color list)
            color_list (list of strings): list of colors for each positions object (default ['#ffff00','#03fc03','#aa00ff'])
            plot_height (integer): height of the plot in pixels (default 400) (width is scaled accordingly)
            size (integer): marker size (default 3)

        Returns:
            overlain holoviews plots
    """
    if len(positions_list) > len(color_list):
        raise Exception("More alignments than colors. Please provide more colors.")
    yaxis_label = '-'.join(matrix.deflines1)
    xaxis_label = '-'.join(matrix.deflines0)
    im = [imshow(matrix,plot_height=plot_height)]
    scatter_list = []
    for index, positions in enumerate(positions_list):
        if (positions.deflines0 != matrix.deflines0) or (positions.deflines1 != matrix.deflines1):
            raise Exception('The alignment ('+'-'.join(positions.deflines0)+' vs. '+'-'.join(positions.deflines1)+') does not match the matrix ('+xaxis_label+' vs. '+yaxis_label+').')
        scatter_list.append(plot_positions(positions,size=size).opts(color=color_list[index]))
    return hv.Overlay(im + scatter_list)


def histogram(df,column,bincount,labelname,color):
    """ Plots a histogram.

        Parameters:
            df (pandas dataframe): data
            column (column name): column of values in data to plot
            bincount (integer): number of bins to use
            labelname (string): name of data you're plotting
            color (string): color for bars in histogram

        Returns:
            holoviews histogram
    """
    frequencies, edges = np.histogram(df[column].tolist(), bincount,density=True)
    print('Values: %s, Edges: %s' % (frequencies.shape[0], edges.shape[0]))
    return hv.Histogram((edges, frequencies), label=labelname).opts(color=color,width=1000, height=500, alpha=0.7,xlabel=column,ylabel='Density')


def multihistogram(df,column,bincount,groupby,color_col):
    """ Plots multiple histograms.

        Parameters:
            df (pandas dataframe): data
            column (column name): column of values in data to plot
            bincount (integer): number of bins to use
            groupby (list): values to group the data by to create multiple histograms
            color_col (column name): column in dataframe specifying the color of each groupby category

        Returns:
            overlain holoviews histograms
    """
    binsize = (df[column].max()-df[column].min())/bincount
    grouped = df.groupby(groupby)
    dfs = [group for _, group in grouped]
    plots = []
    for groupdf in dfs:
        if len(groupdf)>0:
            binno = int((groupdf[column].max()-groupdf[column].min())/binsize)+1
            group_color = groupdf[color_col].tolist()[0]
            plots.append(histogram(groupdf,column,binno,groupdf[groupby].tolist()[0],group_color))#.opts(clabel=groupdf[groupby].tolist()[0]))
    return hv.Overlay(plots)


def distribution(df,column,labelname, color):
    """ Plots a distribution.

        Parameters:
            df (pandas dataframe): data
            column (column name): column of values in data to plot
            labelname (string): name of data you're plotting
            color (string): color for bars in histogram

        Returns:
            holoviews distribution
    """
    return hv.Distribution(df[column].tolist(),label=labelname).opts(color = color)

def multidistribution(df,column,groupby,color_col):
    """ Plots multiple distributions.

        Parameters:
            df (pandas dataframe): data
            column (column name): column of values in data to plot
            groupby (list): values to group the data by to create multiple histograms
            color_col (column name): column in dataframe specifying the color of each groupby category

        Returns:
            overlain holoviews distributions
    """
    grouped = df.groupby(groupby)
    dfs = [group for _, group in grouped]
    plots = []
    for groupdf in dfs:
        if len(groupdf)>0:
            groupdf_range = groupdf[column].max()-groupdf[column].min()
            group_color = groupdf[color_col].tolist()[0]
            plots.append(distribution(groupdf,column,groupdf[groupby].tolist()[0],group_color))#.opts(clabel=groupdf[groupby].tolist()[0]))
    return hv.Overlay(plots)


def multiscatter(df,x,y,groupby,colors_col):
    """ Plots a grouped scatterplot.

        Parameters:
            df (pandas dataframe): data
            x (column name): column to plot on the x axis
            y (column name): column to plot on the y axis
            groupby (column name): column in dataframe with values on which to group the points
            color_col (column name): column in dataframe specifying the color of each groupby category

        Returns:
            holoviews scatterplot
    """
    new_col_order = [x,y] + [i for i in list(df.columns) if i not in [x,y]]
    return hv.Points(data=df[new_col_order],vdims=[groupby,colors_col]).opts(alpha=.5,color=colors_col)


def gap_function(function):
    """ Visualizes a gap function up to length 20.

        Parameters:
            function (gap function): gap function that takes as it's only argument the length of the gap

        Returns:
            holoviews curve plot
    """
    points = [(i, function(i)) for i in range(21)]
    return hv.Curve(points)

def plot_offset_correlations(table,color):
    """ Plots fft offset correlations.

        Parameters:
            table (numpy array): table of offsets and correlations

        Returns:
            holoviews plot
    """
    offsets = table[:,1]
    correlations = table[:,0]
    scatter = hv.Scatter((offsets, correlations)).opts(color=color)
    spikes = hv.Spikes(scatter).opts(color=color)
    return (spikes * scatter).opts(xlabel='offset',ylabel='correlation')



def _percentile(n):
    def percentile_(x):
        return abs(x.mean() - x.quantile(n))
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_


def plot_homstrad_performance(df,groupby):
    """ Plots performances against the HOMSTRAD benchmark, grouped by whatever you'd like.

        Parameters:
            df (pandas dataframe): results of benchmark.homstrad_benchmark_tools_against_ref()
            groupby (column): column of dataframe to group results by

        Returns:
            holoviews plot
    """
    reference = df['reference'].tolist()[0]
    calculated_stats = df.groupby(groupby).agg(
        {"sensitivity": [np.mean, np.max, np.min, _percentile(.75), _percentile(.25), _percentile(.95), _percentile(.05)],
         "precision": [np.mean, np.max, np.min, _percentile(.75), _percentile(.25), _percentile(.95), _percentile(.05)],
         "reference": lambda x: x.iloc[0]}
    ).reset_index()
    calculated_stats.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in calculated_stats.columns]
    calculated_stats.rename(columns={'tool_':'tool','family_':'family'})
    axis_max = max(calculated_stats['sensitivity_mean'].max(),calculated_stats['precision_mean'].max())+.01
    axis_min = min(calculated_stats['sensitivity_mean'].min(),calculated_stats['precision_mean'].min())-.01
    plots = []
    points = hv.Points(
        data = df,
        kdims = ['precision_mean','sensitivity_mean'],
        vdims = [groupby],
    ).opts(
        legend_position='top_left',
        tools=['hover'],
        title = 'Performance w.r.t. ' + str(reference) + ', grouped by ' + str(groupby)
    )
    return points

