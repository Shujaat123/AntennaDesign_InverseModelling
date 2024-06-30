import matplotlib.pyplot as plt
import numpy as np
def plots_a(q,test_dec,test_inputs,test_en,test_targets):
  yyxx=np.linspace(400,2000,102)
  plt.subplot(1, 3, 1)
  plt.imshow(test_dec[q, :].squeeze(0).detach().cpu().numpy())
  plt.subplot(1, 3, 2)
  plt.imshow(test_inputs[q, :].squeeze(0).detach().cpu().numpy())
  plt.subplot(1, 3, 3)
  plt.plot(yyxx, test_en[q, :].squeeze(0).detach().cpu().numpy())
  plt.plot(yyxx, test_targets[q, :].squeeze(0).detach().cpu().numpy())
  plt.title('test')
  plt.show()
