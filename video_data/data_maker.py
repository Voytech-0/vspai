import click
import pandas as pd
import numpy as np
import os
import decord
import multiprocessing


def one_core_remove_corrupted(split):
    split = list(split)
    for filename in split[:]:
        try:
            decord.VideoReader(filename, ctx=decord.cpu(0))
        except Exception as e:
            print(f"Corrupted file: {filename}, Error: {e}")
            split.remove(filename)
    return np.asarray(split)


def remove_corrupted(df: pd.DataFrame):
    n_cores = os.cpu_count()
    df_np = df['image'].to_numpy()
    df_split = np.array_split(df_np, n_cores)

    # Create a pool of workers
    pool = multiprocessing.Pool(n_cores)

    # Map the function to the data chunks
    results = pool.map(one_core_remove_corrupted, df_split)

    pool.close()
    pool.join()


    # Combine the results
    df_np = np.concatenate(results)

    # Update the DataFrame
    df = df[df['image'].isin(df_np)]

    return df


@click.group()
def cli() -> None:
    pass


@cli.command(name='make_csv')
@click.option('--data-folder', type=click.Path(exists=True, file_okay=False), required=True)
@click.option('--output-folder', type=click.Path(file_okay=False), required=False, default='.')
@click.option('--fake', is_flag=True, default=False)
@click.option('--split', type=click.Choice(['train', 'val', 'test']), default='train')
@click.option("--rm-corrupted", is_flag=True, default=False)
@click.option("--file-number-limit", type=int, default=10100)
def make_csv(data_folder, output_folder, fake, split, rm_corrupted, file_number_limit):
    output_name = 'fake_' if fake else 'real_'
    if '/' not in data_folder:
        output_name += data_folder + '.csv'
    else:
        output_name += data_folder.split('/')[-1] + '.csv'
    # create df with  image,class,split headers
    df = pd.DataFrame(columns=['image', 'class', 'split'])
    class_value = 1 if fake else 0
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            filename = (os.path.join(root, file))
            # add row to df
            if filename.endswith('.mp4'):
                df = pd.concat([df, pd.DataFrame({'image': [filename], 'class': [class_value], 'split': [split]})])

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    if file_number_limit is not None:
        df = df[:file_number_limit]

    if rm_corrupted:
        df = remove_corrupted(df)

    df.to_csv(os.path.join(output_folder, output_name), index=False)


if __name__ == '__main__':
    cli()
