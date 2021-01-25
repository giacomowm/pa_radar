# Proportional Area Radar Chart Use Guide
The file `pa_radar` to plot radar charts whose enclosed area is proportional to the sum (or average) of the values (when normalised).

## Simple First Use
Ensure the file `pa_radar.p` is in you working directory. We need to load the class of the same name (`pa_radar`) to set up a plot.
```
from pa_radar import pa_radar
```

We can then create and object that we can use for plotting. At first, without choosing any options this can be done most simply by
```
my_radar = pa_radar()
```

We can then plot some data (in percentile format for now, i.e. values between 0 and 1) by

```
my_radar.plot([0.5,0.1,0.8,0.3,1])
```

A plot of two (or more) datasets can be overlayed by providing both in a list (note the extra square brackets). Note these must be of the same length.

```
my_radar.plot([[0.5,0.1,0.8,0.3,1],[0.3,0.7,0.3,0.2,0.1]])
```

The above two lines produce the following two plots:

![default plots](/images/default_plot.png)

We are implicitly using the `percentiles` variable here and so 

```
my_radar.plot(percentiles = [0.5,0.1,0.8,0.3,1])
```
is equivalent to the first plot.

To plot values that do not lie in the range 0,1 (and the change the axis values accordingly) we instead use the `values` parameter. We then need to define `ranges` for our data. An example shows how to do this:

```

```


