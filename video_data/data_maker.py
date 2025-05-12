import click
import pandas as pd
import os


@click.group()
def cli() -> None:
    pass


@cli.command(name='make_csv')
@click.option('--data-folder', type=click.Path(exists=True, file_okay=False), required=True)
@click.option('--output-folder', type=click.Path(file_okay=False), required=False, default='.')
@click.option('--fake', is_flag=True, default=False)
@click.option('--split', type=click.Choice(['train', 'val', 'test']), default='test')
def make_csv(data_folder, output_folder, fake, split):
    output_name = 'fake_' if fake else 'real_'
    output_name += data_folder + '.csv'
    # create df with  image,class,split headers
    df = pd.DataFrame(columns=['image', 'class', 'split'])
    class_value = 1 if fake else 0
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            filename = (os.path.join(root, file))
            # add row to df
            df = pd.concat([df, pd.DataFrame({'image': [filename], 'class': [class_value], 'split': [split]})])
    df.to_csv(os.path.join(output_folder, output_name), index=False)


if __name__ == '__main__':
    cli()
