import numpy as np
import matplotlib.pyplot as plt
import flow_vis


def visualize_results(rgb_pair: tuple, depth_pair: tuple, flow: np.ndarray,
    rigid_flo: np.ndarray, nonrigid_flo: np.ndarray, ds_name: str):

    rgb1, rgb2 = rgb_pair
    depth1, depth2 = depth_pair

    fig = plt.figure(figsize=(6.5, 5 * 6.5))

    gs = fig.add_gridspec(12, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[3, 0])
    ax7 = fig.add_subplot(gs[3, 1])
    ax8 = fig.add_subplot(gs[4, 0])
    ax9 = fig.add_subplot(gs[4, 1])
    ax10 = fig.add_subplot(gs[5:7, :])
    ax11 = fig.add_subplot(gs[7:9, :])
    ax12 = fig.add_subplot(gs[9:11, :])
    ax13 = fig.add_subplot(gs[11, 0])
    ax14 = fig.add_subplot(gs[11, 1])

    ax0.imshow(rgb1)
    ax0.set_title('1st frame')
    ax0.set_xticks([]); ax0.set_yticks([])
    ax1.imshow(rgb2)
    ax1.set_title('2nd frame')
    ax1.set_xticks([]); ax1.set_yticks([])
    ax2.imshow(np.log(depth1))
    ax2.set_title('log depth')
    ax2.set_xticks([]); ax2.set_yticks([])
    ax3.imshow(np.log(depth2))
    ax3.set_title('log depth')
    ax3.set_xticks([]); ax3.set_yticks([])

    ax4.matshow(flow[..., 0])
    ax4.set_title('gt flow: u')
    ax4.set_xticks([]); ax4.set_yticks([])
    ax5.matshow(flow[..., 1])
    ax5.set_title('gt flow: v')
    ax5.set_xticks([]); ax5.set_yticks([])

    ax6.matshow(rigid_flo[..., 0])
    ax6.set_title('rigid flow: u')
    ax6.set_xticks([]); ax6.set_yticks([])
    ax7.matshow(rigid_flo[..., 1])
    ax7.set_title('rigid flow: v')
    ax7.set_xticks([]); ax7.set_yticks([])

    ax8.matshow(nonrigid_flo[..., 0])
    ax8.set_title('nonrigid flow: u')
    ax8.set_xticks([]); ax8.set_yticks([])
    ax9.matshow(nonrigid_flo[..., 1])
    ax9.set_title('nonrigid flow: v')
    ax9.set_xticks([]); ax9.set_yticks([])

    # visualize all types of optical flow with common scale
    flow_viz, rigid_flo_viz, nonrigid_flo_viz = np.split(
        flow_vis.flow_to_color(
            np.concatenate((flow, rigid_flo, nonrigid_flo), axis=0),
            convert_to_bgr=False), 3, axis=0
    )

    ax10.imshow(flow_viz)
    ax10.set_title('ground truth optical flow')
    ax10.set_xticks([]); ax4.set_yticks([])
    ax11.imshow(rigid_flo_viz)
    ax11.set_title('optical flow due to camera motion (rigid flow)')
    ax11.set_xticks([]); ax5.set_yticks([])
    ax12.imshow(nonrigid_flo_viz)
    ax12.set_title('residual flow after subtracting flow due to camera motion (nonrigid flow)')
    ax12.set_xticks([]); ax6.set_yticks([])

    ax13.scatter(rigid_flo[..., 0].ravel(), flow[..., 0].ravel(), c='k',
                alpha=.05, s=1)
    ax13.set_xlabel('rigid flow')
    ax13.set_ylabel('GT')
    ax13.set_title('u')
    ax13.plot(flow[..., 0].ravel(), flow[..., 0].ravel(), c='g', linestyle=':')
    ax14.scatter(rigid_flo[..., 1].ravel(), flow[..., 1].ravel(), c='k',
                alpha=.05, s=1)
    ax14.set_xlabel('rigid flow')
    ax14.set_ylabel('GT')
    ax14.set_title('v')
    ax14.plot(flow[..., 1].ravel(), flow[..., 1].ravel(), c='g', linestyle=':')

    plt.suptitle(ds_name.upper())
    plt.tight_layout()
