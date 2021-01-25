#Proportional Area Radar Charts
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
This produces:

![GitHub Logo](/images/image.png)

A plot of two (or more) datasets can be overlayed by providing both in a list (note the extra square brackets). Note these must be of the same length.

```
my_radar.plot([[0.5,0.1,0.8,0.3,1],[0.3,0.7,0.3,0.2,0.1]])
```
