# myapp.py

from random import random
import numpy as np

from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

from q2_mlab.plotting.components import (
    Mediator,
    Plottable,
    ButtonComponent,
    TextPlotComponent,
    ScatterComponent,
    SegmentComponent,
)


i = 0


class RandomPositionTextPlot(Mediator, Plottable):

    def __init__(self):
        super().__init__()
        self.i = 0
        self.button = None
        self.text_plot = None
        self.scatter = None
        self.segment = None
        self.layout = None

    def notify(self, component, event_name,
               event=None,
               attrname=None,
               old=None,
               new=None,
               ):
        if event_name == 'button-click':
            self.increment_text_plot()

    def increment_text_plot(self):
        self.text_plot.add_text(
            x=[random() * 20 + 15],
            y=[random() * 70 + 15],
            text=[str(self.i)],
            text_color=[RdYlBu3[self.i % 3]],
        )
        self.i = self.i + 1

    def plot(self):
        figure_kwargs = {'x_range': (0, 100), 'y_range': (0, 100),
                         'toolbar_location': None
                         }
        text_kwargs = {
            'x': [],
            'y': [],
            'text': [],
            'text_color': [],
            'text_font_size': "26px",
            'text_baseline': 'middle',
            'text_align': 'center',
        }
        scatter_data = {
            'x': [10, 20, 30, 40, 50, 60, 70, 80],
            'y': [10, 20, 30, 40, 50, 60, 70, 80],
        }
        scatter_source = ColumnDataSource(scatter_data)

        N = 9
        x = np.linspace(-2, 2, N)
        y = x ** 2

        def move(x):
            return (x + 2) * 20

        segment_source = ColumnDataSource(dict(
            x=move(x),
            y=move(y),
            xm01=move(x - x ** 3 / 10 + 0.3),
            ym01=move(y - x ** 2 / 10 + 0.5),
        ))

        self.button = ButtonComponent(button_kwargs={'label': "Press Me"})
        self.button.set_mediator(self)
        self.text_plot = TextPlotComponent()
        self.text_plot.set_mediator(self)
        self.text_plot.plot(figure_kwargs=figure_kwargs,
                            text_kwargs=text_kwargs
                            )
        self.text_plot.layout.border_fill_color = 'black'
        self.text_plot.layout.background_fill_color = 'black'
        self.text_plot.layout.outline_line_color = None
        self.text_plot.layout.grid.grid_line_color = None

        self.scatter = ScatterComponent(
            data_source=scatter_source,
        )
        scatter_kwargs = {
            'x': 'x',
            'y': 'y',
        }
        self.scatter.plot(
            figure=self.text_plot.layout,
            scatter_kwargs=scatter_kwargs,
        )

        self.segment = SegmentComponent(
            data_source=segment_source,
        )
        segment_kwargs = {
            "x0": "x",
            "y0": "y",
            "x1": "xm01",
            "y1": "ym01",
            "line_color": "#f4a582",
            "line_width": 3,
        }
        self.segment.plot(
            figure=self.text_plot.layout,
            segment_kwargs=segment_kwargs,
        )

        self.layout = column(self.button.layout, self.text_plot.layout)

        return self

    def app(self, doc):
        doc.add_root(self.layout)


def app(doc):
    # create a plot and style its properties
    p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
    p.border_fill_color = 'black'
    p.background_fill_color = 'black'
    p.outline_line_color = None
    p.grid.grid_line_color = None

    # add a text renderer to our plot (no data yet)
    r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="26px",
               text_baseline="middle", text_align="center")


    ds = r.data_source

    # create a callback that will add a number in a random location
    def callback():
        global i

        # BEST PRACTICE --- update .data in one step with a new dict
        new_data = dict()
        new_data['x'] = ds.data['x'] + [random() * 20 + 15]
        new_data['y'] = ds.data['y'] + [random() * 70 + 15]
        new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i % 3]]
        new_data['text'] = ds.data['text'] + [str(i)]
        ds.data = new_data

        i = i + 1

    # add a button widget and configure with the call back
    button = Button(label="Press Me")
    button.on_click(callback)

    # put the button and plot in a layout and add to the document
    doc.add_root(column(button, p))


if __name__.startswith('bokeh_app'):
    plot = RandomPositionTextPlot()
    plot.plot()
    plot.app(curdoc())
    # app(curdoc())
