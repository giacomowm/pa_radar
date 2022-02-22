# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:38:46 2020

@author: Giacomo Marchesi
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import integrate
from sympy import solveset
from sympy import Eq
# from sympy import sympify
# from sympy import Abs
import copy


class pa_radar():
    """
    Class for setup and plotting of Proportional Area Radar Charts
    """
        
    def __init__(self,offset=0.5,proportion=15/4,
                 centre_proportion = 1,
                 team_name = None,
                 ranges = None,
                 names = None,
                 categories = None,
                 text_sizes = None,
                 background_colour = None,
                 counter_colour=None,
                 backing_colours = None,
                 colours = None,
                 alphas = None,
                 title = None,
                 notes = None,
                 attribution = None,
                 mode = None,
                 axes = True,
                 axes_intervals = None,
                 backing= True,
                 backing_intervals = None,
                 boxes = None,
                 box_rotation = True,
                 scores = None,
                 numbers = None,
                 bar = None,
                 bar_mode = None,
                 linear = None,
                 flower = None,
                 polygonal_centre = True,
                 key = None,
                 font = None,
                 text_colour = None,
                 bar_outline = False,
                 clock = 1,
                 hive = False,
                 discontinuous = False,
                 step=20):
        self.offset = offset
        self.team_name = team_name
        self.ranges = ranges
        self.names = names
        self.categories = categories
        self.text_colour = text_colour
        self.text_sizes = text_sizes
        self.background_colour = background_colour
        self.counter_colour = counter_colour
        self.backing_colours = backing_colours
        self.title = title
        self.notes = notes
        self.attribution = attribution
        self.mode = mode
        self.axes = axes
        self.axes_intervals = axes_intervals
        self.backing = backing
        self.backing_intervals = backing_intervals
        self.boxes = boxes
        self.box_rotation = box_rotation
        self.scores = scores
        self.linear = linear
        self.key = key
        self.font=font
        self.text_colour = text_colour
        self.bar_outline = bar_outline
        self.radar_offset = offset
        self.radar_proportion = proportion
        self.bar = bar
        self.bar_mode = bar_mode
        self.flower=flower
        self.clock = clock
        self.hive = hive
        self.discontinuous = discontinuous
        self.step = step
        
        
        # Set defaults for unassigned attributes
        self.colours = colours
        self.alphas = alphas
        if team_name is not None:
            colours,alphas = self._team_colours(team_name)
            if self.colours is None:
                self.colours = colours
            if self.alphas is None:
                self.alphas = alphas
       
            
        # MATHS HERE
        # Declare algebraic symbols:
        W = sym.Symbol('W')
        A = sym.Symbol('A')
        B = sym.Symbol('B')
        a = sym.Symbol('a')
        n = sym.Symbol('n')
        t = sym.Symbol('t')
        x = sym.Symbol('x')
        half = sym.Rational(1, 2)
        #Choose mode and define and (if necessary) solve equations
        if self.bar == True:
            self.radar_equation = A + offset + 0*(B+n+t)
            if self.bar_mode == 'area':
                self.radar_equation = sym.sqrt(offset**2 + A*(1 + 2*offset))
        elif linear == True:
            self.radar_equation = (A + offset)*(B+offset)*sym.sin(2*np.pi/n)/((sym.sin(2*np.pi/n))*(B+offset)*sym.cos(t) + (A+offset-(B+offset)*sym.cos(2*np.pi/n))*sym.sin(t))
        
        elif polygonal_centre == True: # for a non-circular counter
            # Define the interpolation curve
            W = a*(-sym.sqrt(A)*(1-x)+sym.sqrt(B)*x)**2 + (1-a)*(1-(-sym.sqrt(1-A)*(1-x)+sym.sqrt(1-B)*x)**2 )  
            # Add offset (central plot counter radius)
            F = W + offset + offset*(x-1)*x         
            # Transform to polar coordinates in the range 0 to 2pi/n
            F = F.subs(x,n*t/(2*sym.pi))
            # Integrate
            G = integrate(half*F**2, (t, 0, 2*sym.pi/n))
            # Solve for proportional area
            solutions = solveset(Eq(G, (A+B)/2*proportion/n*(1-1/n)*1.15 + centre_proportion*(1-1/n+1/20 + 8/((n)**2))*(offset**2)*sym.sin(np.pi/n)*sym.cos(np.pi/n)),a)
            # SOL =       solveset(Eq(G,(A+B)/2*proportion/n*(1-1/n)*1.15 + (1-1/n+1/20 + 8/((n)**2))*(offset**2)*sym.sin(np.pi/n)*sym.cos(np.pi/n)),a)
            # Retrieve the right solution (of the two)
            if offset == 1:
                sol = solutions.args[0]
            else:
                sol = solutions.args[1]
            # Substitute value for a into original equation
            H = F.subs(a,sol)
            # set this equation to be used by radar_plot
            self.radar_equation = H
        else: # the default setting
            # Define the interpolation curve
            W = a*(-sym.sqrt(A)*(1-x)+sym.sqrt(B)*x)**2 + (1-a)*(1 - (-sym.sqrt(1-A)*(1-x) + sym.sqrt(1-B)*x)**2 )
            # Add offset (central plot counter radius)
            F = W + offset
            # Transform to polar coordinates in the range 0 to 2pi/n
            F = F.subs(x,n*t/(2*sym.pi))
            # Integrate
            G = integrate(half*F**2, (t, 0, 2*sym.pi/n))
            # Solve for proportional area
            solutions = solveset(Eq(G,(A+B)/2*proportion/n + sym.pi/n*offset**2),a)
            # Retrieve the right solution (of the two)
            sol = solutions.args[0]
            # Substitute value for a into original equation
            H = F.subs(a,sol)
            # Declare global values to be used by radar_plot
            self.radar_equation = H
        
            
    # Function for checking if one or many players to be displayed 
    def _check_multiple(self,player_s):
        if type(player_s[0]) is list:
            multiple = True
            if len(player_s[0]) == 2:
                multiple = False
        else:
            multiple = False
        return multiple

    
    # Function for creating the values for the filled curves
    def _get_values(self,values,discontinuous):
        
        N = len(values)
        
        
        disc = 1
        if discontinuous is True:
            #print('discontinuous')
            N =int(N/2)
            disc = 2
        
        # Declare algebraic symbols:
        A = sym.Symbol('A')
        B = sym.Symbol('B')
        n = sym.Symbol('n')
        t = sym.Symbol('t')
        # retrieve radar equation and set n = number of values.
        H = self.radar_equation.subs(n,N)
        # Create vectors for
        
        
        
        
        
        thetas = [0]*N
        radii = [0]*N
        if self.bar is True:
            bar_correct = 0.5
        else:
            bar_correct = 0
        
        
        evaluation_func =  lambda z: H.subs([(A,values[j]),(B,values[(j+1)%len(values)]),(t,(z-2*i*sym.pi/N))]).evalf()
        evaluation = np.vectorize(evaluation_func)
        for i in range(N):
            j = i
            thetas[i] = np.arange((i-bar_correct)*2*np.pi/N,(i+1-bar_correct)*2*np.pi/N+1*np.pi/(N*self.step),2*np.pi/(N*self.step))
            radii[i] = thetas[i]
        for i in range(N):
            j=i*disc
            radii[i] = np.array(evaluation(thetas[i]),dtype = float)
        theta = np.concatenate(thetas)
        radius = np.concatenate(radii)
        pi_2 = np.array([2*np.pi*(1 - bar_correct/N)])
        theta = np.append(theta,pi_2)
        radius = np.append(radius,radius[0])
        return theta, radius
  
    
    
    def _upright_angle(self,angle):
        """
        Function for making angles rotated beyond a half turn upright again
        

        Parameters
        ----------
        angle : anglue in dgrees

        Returns
        -------
        angle : angle in degrees

        """
        if (angle%360 > 90) and (angle%360 < 270):
            angle = angle + 180
        return angle

    
    
    def _my_rounder(self,x):
        """
        Function for formatting numbers

        Parameters
        ----------
        x : float

        Returns
        -------
        w : formatted string

        """
        # #print(x)
        y = float('{0:g}'.format(x))
        # #print('y=',y)
        z=y
        if -1<y<1:
            z= float('{0:.2g}'.format(y))
        if y < -10 or y > 10:
            z = float('{0:.2f}'.format(y))
        # #print('z=',z)
        w='{0:.3g}'.format(z)
        # #print('final:', w)
        return w
    
    def _interp(self,value, range):
        """
        linear interpolator that copes with 'backwards' ranges (ie range[0] > range[1])

        Parameters
        ----------
        value :  values
        range :  ranges

        Returns
        -------
        value interpolated in range
        """
        
        inner = range[0]
        outer = range[1]
        if isinstance(value, list):
            #print('listAhoy')
            return [(value[0] - inner) / (outer - inner),(value[1] - inner) / (outer - inner)]
        else:
            return (value - inner) / (outer - inner)
    
        
    def plot(self, plot_data = None,
                   plot_data_name = None,
                   values = None,
                   values_name = None,
                   ranges = None,
                   names = None,
                   categories = None,
                   text_sizes = None,
                   background_colour = None,
                   counter_colour = None,
                   backing_colours = None,
                   colours = None,
                   alphas = None,
                   title = None,
                   notes = None,
                   attribution = None,
                   axes = None,
                   axes_intervals = None,
                   backing = None,
                   backing_intervals = None,
                   team_name = None,
                   mode = None,
                   boxes = None,
                   box_pos = None,
                   box_rotation = None,
                   scores = None,
                   flower = None,
                   key = False,
                   font = None,
                   text_colour = None,
                   bar_outline = None,
                   clock = None,
                   discontinuous = None
                   ):
        
        # Get fixed variables from radar_plot_setup()
        offset = self.radar_offset
        
        # Get defaults for optional variables not defined
        if ranges is None:
            ranges = self.ranges
        if categories is None:
            categories = self.categories
        if names is None:
            names = self.names
        if text_colour is None:
            text_colour = self.text_colour
        if text_sizes is None:
            text_sizes = self.text_sizes
        if background_colour is None:
            background_colour = self.background_colour
        if counter_colour is None:
            counter_colour = self.counter_colour 
        if backing_colours is None:
            backing_colours = self.backing_colours
        if colours is None:
            colours = self.colours
        if alphas is None:
            alphas = self.alphas
        if title is None:
            title = self.title
        if notes is None:
            notes = self.notes
        if attribution is None:
            attribution = self.attribution
        # if attribution is None:
        #     attribution = 'pa_radar @GiacomoWM'
        if team_name is None:
            team_name = self.team_name
        if mode is None:
            mode = self.mode
        if axes is None:
            axes = self.axes
        if axes_intervals is None:
            axes_intervals = self.axes_intervals
        if backing is None:
            backing = self.backing
        if backing_intervals is None:
            backing_intervals = self.backing_intervals
        if boxes is None:
            boxes = self.boxes
        if box_rotation is None:
            box_rotation = self.box_rotation
        if scores is None:
            scores = self.scores
        if flower is None:
            flower = self.flower
        if key is None:
            key = self.key
        if font is None:
            font = self.font
        if text_colour is None:
            text_colour = self.text_colour
        bar = self.bar
        if bar_outline is None:
            bar_outline = self.bar_outline
        if clock is None:
            clock = self.clock
        if discontinuous is None:
            discontinuous = self.discontinuous
        #print(bar)
        
        ##
        # safety !!!
        data = copy.deepcopy(plot_data)
        
        
        #make clock positive clockwise
        clock = -clock

        
        # Set defaults for unassigned attributes
        
        if backing_colours == None:
            grey_1 = 0.84
            grey_2 = 0.95
            backing_colours = [(grey_1,grey_1,grey_1),(grey_2,grey_2,grey_2)]
        
        if background_colour is None:
            background_colour = 'white'
        if counter_colour is None:
            counter_colour = background_colour
        
        if colours is None:
            colours = ['red','blue','green','black']
        if alphas is None:
            alphas = [.6,.45,.4,.3]
        
        #print ('making radar...')
        
        # Count the number of categoreis and number of players
        multiple = False # default
        if data is not None:
            multiple = self._check_multiple(data)
        if values is not None:
            multiple = self._check_multiple(values)
        
        

        if multiple is True:
            if data is not None:
                n_data = len(data[0])
                p_data = len(data)
            else:
                n_data = 0 
                p_data = 0
            if values is not None:
                n_val = len(values[0])
                p_val = len(values)
            else:
                n_val = 0
                p_val = 0
        else:
            if data is not None:
                n_data = len(data)
                p_data = 1
            else:
                n_data = 0
                p_data = 0
            if values is not None:
                n_val = len(values)
                p_val = 1
            else:
                n_val = 0
                p_val = 0
        if discontinuous is True:
                    n_data = int(n_data/2)
                    disc = 2
        else:
            disc = 1
        if ranges is not None:
            n_rang = len(ranges)
        else:
            n_rang = 0
        if categories is not None:
            n_cat = len(categories)
        else:
            n_cat = 0
            
        n = max(3,n_data,n_val,n_rang,n_cat) # number of categories
        #print (n)
        p = max(1,p_data,p_val) # number of players
        #print('number of categories is ',n)
        #print('number of objects is ',p)
        # Make single list data a list of one list
        if p == 1:
            if data is not None:
                data = [data]
            if values is not None:
                values = [values]
        if ranges is not None:
            #print(ranges[0])
            if isinstance(ranges[0],int) or isinstance(ranges[0],float):
                ranges = [ranges]
        #print('ranges = ',ranges)
        
        
        # Set mode:
        if mode is None:
            mode = 'percentile'
            if ranges is not None:
                mode = 'value'
            # else:
                ##print("Using default mode: 'percentile'. Data must be in range 0 to 1. Axes will display on a scale 0 to 100. For other data or axes add an explicit ranges parameter.")
        ##print('mode: ', mode)
        ##print('boxes: ', boxes)
        ##print('scores: ', scores)
        
        # Set data
        if ranges is not None and data is not None: # then normalise the values
            ##print('ranges not none')
            for i in range(len(data)):
                for j in range(len(data[i])):
                    data[i][j] = self._interp(data[i][j],ranges[int(int(j)%(len(ranges)))]) # Normalise data
            #print('data set to custom ranges')
                            # #print('values3',values)
        if data is None:
            # data = [[0.0]*(n*disc)]
            print('No data')
            
        ###########################################################################
        # Set up plot
        fig = plt.figure(figsize =(5,5),dpi=600)
        fig.set_tight_layout(True)
        fig.patch.set_facecolor(background_colour)
        ax = plt.subplot(111, projection='polar')
        ax.set_facecolor(background_colour)
        
        ###Text sizes
        if text_sizes is None:
            text_sizes = [6,8,10,12,16]
        tiny_text = text_sizes[0]
        small_text = text_sizes[1]
        medium_text = text_sizes[2]
        large_text = text_sizes[3]
        title_text = text_sizes[4]
        
        
        #Set up backing and axes intervals
        if axes_intervals is None:
                axes_intervals = 4
        if backing_intervals is None:
            backing_intervals = axes_intervals
        
        
        #label axes
        if axes is True:
            pos = np.arange(0, 1, 1/axes_intervals )
            pos = np.append(pos,[1])
            # pos = [offset,offset+0.2,offset+0.4,offset+0.6,offset+0.8,offset+1]# Less radial ticks
            labels = []
            for i in range(n):
                labels.append(copy.deepcopy(pos))
            if (mode != 'percentile') and (ranges is not None):
                for i in range (n):
                    for j in range(len(pos)):
                        labels[i][j] = np.interp(pos[j],[0,1],ranges[i%len(ranges)])
            if mode == 'percentile':
                scale_axis = 100
            else:
                scale_axis = 1
            #print(labels)
            if (self.bar is True) and (self.bar_mode == 'area'):
                #print('bar area adjust')
                for i in range(len(pos)):
                    pos[i] =  sym.sqrt(offset**2 + (pos[i])*(1 + 2*offset))
            else:
                pos = pos + offset
            for i in range(n):
                
                for j in range (len(pos)):
                    ax.annotate(self._my_rounder(labels[i][j]*scale_axis), 
                                font=font,
                                fontsize = tiny_text,
                                xy=[clock*(i*2*np.pi/n), pos[j]],
                                xycoords='data',
                                rotation = self._upright_angle(clock*360*i/n),
                                ha="center",
                                va="center",
                                color = text_colour,
                                zorder = backing_intervals+3+p+1)
                
        # Plot chart backing as level sets in multiples of 0.2     
        # Set overlapping for layers
        zorders = np.arange(1,backing_intervals+1,1)
        pos = np.arange(0,  1, 1/backing_intervals )
        pos = np.append(pos,[1])
        # Draw the chart backing
        if backing is True:
            #print('plotting backing')
            for i in range(backing_intervals):
                #print('plotting backing curve', i+1, 'of', backing_intervals)
                level = [pos[-i-1]]*n*disc
                if flower is True:
                    level = [pos[-i-1],0]*n
                theta,radius = self._get_values(level,discontinuous)
                theta = theta*clock
                ax.plot(theta, radius,alpha=0)
                ax.fill_between(theta, radius,alpha = 1,color = backing_colours[i%len(backing_colours)],lw=0,zorder = zorders[i])
            
        #Draw the central counter
        if flower is True:
            level = [0]*(2*n)*disc
        else:
            level = [0]*n*disc
        theta,radius = self._get_values(level,discontinuous)
        ax.plot(theta, radius,alpha=0)
        w = [ax.fill_between(theta, radius,alpha = 1,color = counter_colour,lw=0,zorder = backing_intervals+3+p)]
             
        
        #plot each player
        if data is not None:
            #print('plotting filled data curve(s)')
            #print('self.hive = ', self.hive)
            if self.hive is True:
                #print('hive mode')
                for i in range(p):
                    #print( i+1,'of', p)
                    #print(data[i])
                    #print(list(map (lambda x: x[0], data[i])))
                    theta,radius = self._get_values(list(map (lambda x: x[0], data[i])),discontinuous)
                    theta,radius_2 = self._get_values(list(map (lambda x: x[1], data[i])),discontinuous)
                    theta = theta*clock
                    ax.plot(theta,radius_2,radius,alpha=0,lw=0,color = counter_colour,zorder = backing_intervals+2+i)
                    if True: ## to do: different fill colours for underlap vs overlap option here False
                        ax.fill_between(theta, radius,radius_2,alpha = alphas[i%len(alphas)], color = colours[i%len(colours)],lw=0,zorder = backing_intervals+3+i)
                    else:
                        ax.fill_between(theta, radius,radius_2,alpha = alphas[i%len(alphas)], where=(radius_2 > radius), color = colours[i%len(colours)],lw=0,zorder = backing_intervals+3+i)
                        ax.fill_between(theta, radius,radius_2,alpha = alphas[i%len(alphas)], where=(radius_2 < radius), color = colours[-(i%len(colours))-1],lw=0,zorder = backing_intervals+3+i)
            else:
                for i in range(p):
                    if flower is True:
                        #print('flower')
                        data_0 = []
                        for j in range(len(data[i])):
                            data_0 = np.append(data_0,[data[i][j],0])
                        data[i] = data_0
                    #print( i+1,'of', p)
                    theta,radius = self._get_values(data[i],discontinuous)
                    theta = theta*clock
                    if self.bar is True:
                        if bar_outline is True: # #prints outlines to differentiate bars of equal height
                            for j in range(n):
                                ax.plot(np.concatenate([[0],theta[j*(self.step+1):(j+1)*(self.step+1)+1:1],[0]]), np.concatenate([[0],radius[j*(self.step+1):(j+1)*(self.step+1)+1:1],[0]])
                                        ,alpha = alphas[i%len(alphas)] + (1 - alphas[i%len(alphas)])/2,lw=1,color = colours[i%n],zorder = backing_intervals+2+i)
                    ax.plot(theta, radius,alpha=0,lw=0,color = counter_colour,zorder = backing_intervals+2+i)
                    z = ax.fill_between(theta, radius,alpha = alphas[i%len(alphas)], color = colours[i%n],lw=0,zorder = backing_intervals+3+i)
                    w.append(z)
            #print('plotted filled data curve(s)')
                
               
        
        # Add other labeling
        # Type player names over central counter
        if names is not None:
            for i in range(len(names)):
                ax.annotate(names[i],
                            font=font,
                            xy=[0.5,0.5 +0.04*(len(names)-1)- 0.08*i],
                            xycoords='axes fraction',
                            fontsize = medium_text,
                            ha="center",
                            va="center",
                            color = colours[i],
                            zorder = backing_intervals+3+p+2)
        padding = 0.1       
                
        # BOXES percentile scores (or other) in little boxes at value on each axis
        box_data = None
        box_off = 0
        # #print(boxes)
        
        if boxes is not False:
            if boxes == 'percentiles':
                if data is not None:
                    box_data = copy.deepcopy(data)
            elif boxes == 'data':
                if data is not None:
                    box_data = copy.deepcopy(data)
            elif box_pos is not None:
                box_data = box_pos
            else:
                box_data = None
        if box_data is not None:
            box_number = copy.deepcopy(box_data)
            for j in range(len(box_data)):
                bbox_props = dict(boxstyle="round,pad=0.3", fc=background_colour, ec=colours[j%len(colours)], lw=0.5)
                
                for i in range(len(box_data[j])):
                    if boxes == 'percentiles':
                        box_number[j][i] = str(round(box_data[j][i]*100))
                    else:
                        box_number[j][i] = self._my_rounder(box_data[j][i])
                for i in range(n):
                    if self.bar is True:
                        if self.bar_mode == 'area':
                            box_data[j][i]  = sym.sqrt(offset**2 + (box_data[j][i])*(1 + 2*offset) )-offset
                    if not box_rotation:
                        rot = self._upright_angle(clock*360*i/n)
                    else:
                        rot = 0
                    ax.annotate(box_number[j][i], 
                                font = font,
                                fontsize = tiny_text,
                                xy=[clock*(i*2*np.pi/n +box_off*(1-0.4*box_data[j][i])*(0.12*(p-1)-0.24*j)),box_data[j][i]+offset],
                                xycoords='data',
                                rotation = rot,
                                ha='center',
                                va='center',
                                color = colours[j%len(colours)],
                                zorder = backing_intervals+3+2*p,
                                bbox=bbox_props)
                    
        # SCORES actual values (or other) under category names
        score_data = None
        score_mult = 1
        if scores is not None:
            if scores == 'values':
                    score_data = values
            if scores == 'percentiles':
                if data is not None:
                    score_data = data
                    score_mult = 100
            if scores == 'data':
                score_data = data
        if score_data is not None:
            for j in range(p):
                for i in range(n):
                    ax.annotate(self._my_rounder(score_data[j][i]*score_mult),
                                font = font,
                                xy=[clock*(i*2*np.pi/n + 0.08*(p-1) - 0.16*j)
                                    ,(1.05 + small_text/80 + offset)/(np.cos(0.08*(p-1) - 0.16*j))
                                    ],xycoords='data', # correction to a straight line
                                fontsize = small_text,
                                rotation = self._upright_angle(clock*360*i/n),
                                ha='center',#alignments[p-1][j],
                                va='center',
                                color = colours[j],
                                zorder = backing_intervals+3+p+2)
            padding =  0.15 +small_text/50
                    
        # category names around edge
        if scores:
            cat_height = 1.15+small_text/50# Higher to fit values
        else:
            cat_height = 1.15 # Can be lower
        if categories is not None:   #only if categories not empty
            padding = padding + 0.15 + 1*(cat_height-1)
            for i in range(n):
                ax.annotate(categories[i], 
                            font = font, fontsize = small_text, weight = 'bold',
                            xy=[clock*i*2*np.pi/n,cat_height + offset ],xycoords='data',
                            rotation = self._upright_angle(clock*360*i/n),
                            ha="center", va="center",color = text_colour,zorder = backing_intervals+3+p+2)
           
         # tidy up plot
        ax.set_rmax(1+offset+padding) #set overall size of plot
        ax.set_theta_zero_location("N") #first category at the top
        ax.set_rticks([])  # prevent automatic radial ticks
        ax.set_yticklabels([]) # prevent tickw around the edge
        ax.set_xticks([]) # prevent automatic ticks
        # ax.grid(True) # no grid
        ax.spines['polar'].set_visible(False) # no outer circle
        
        
                    
        # add extra text        
        # add note bottom right
        ax.annotate(notes,
                    font = font,
                    xy=[1,0], xycoords='axes fraction',fontsize = small_text,
                    ha="right", va="bottom",color = text_colour,zorder = backing_intervals+3+p+2)
        # add attribution bottom left
        ax.annotate(attribution,
                    font = font,
                    xy=[0,0], xycoords='axes fraction',fontsize = small_text,
                    ha="left", va="bottom",color = text_colour,zorder = backing_intervals+3+p+2)
        
        # Add title top left
        ax.annotate(title,
                    font = font,
                    fontsize = title_text, weight = 'bold',
                    xy=[0,1], xycoords='axes fraction',
                    ha="left", va="top",color = text_colour ,zorder = backing_intervals+3+2*p)
        
        #Add key top right
        if key:
            if (categories is not None) or (scores is not False and score_data is not None) or (boxes is not False and box_data is not None):
                # Key Title
                key_height = 1
                ax.annotate('key',fontweight = 'bold',
                            font = font,
                            xy=[.92,key_height], xycoords='axes fraction',fontsize = small_text,
                            ha="center", va="top",color = text_colour,zorder = backing_intervals+3+2*p)
                # Make space for new line
                key_height = key_height - 0.005
                
                #Key Categories
                if categories is not None:
                    # Make space for new line
                    key_height = key_height -0.03
                    #annotate
                    ax.annotate('Catergory', 
                                font = font,
                                xy=[.92,key_height], xycoords='axes fraction', fontsize = small_text, weight = 'bold',
                                ha="center", va="top",color = text_colour,zorder = backing_intervals+3+2*p)
                # Key Scores (default = values)
                if scores is not False:
                    if score_data is not None:
                        # Make space for new line
                        key_height = key_height -0.03
                        #annotate
                        ax.annotate(scores, 
                                    font = font,
                                    xy=[.92,key_height], xycoords='axes fraction', fontsize = small_text,
                                    ha="center", va="top",color = colours[0],zorder = backing_intervals+3+2*p)
                    bbox_props = dict(boxstyle="round,pad=0.3", fc=background_colour, ec=colours[0], lw=0.5)
                    
                
                # Key Box Data (default: percentile)
                if boxes is not False:
                    if box_data is not None:
                        # Make space for new line
                        key_height = key_height -0.035
                        #annotate
                        ax.annotate(boxes, 
                                    font = font,
                                    xy=[.92,key_height], xycoords='axes fraction', fontsize = small_text,
                                    ha="center", va="top",color = colours[0],zorder = backing_intervals+3+2*p,
                                    bbox=bbox_props)

        return fig, ax
    
    
    