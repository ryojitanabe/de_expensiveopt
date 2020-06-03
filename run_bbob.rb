# -*- coding: utf-8 -*-

out_folder_names = {"syn_de" => "Syn", "asy_de" => "Asy", "plus_de" => "Plus", "wi_de" => "WI", "sts_de" => "STS"}

["syn_de", "asy_de", "plus_de", "wi_de", "sts_de"].each{|de_alg|
  ["hand_tuned", "smac_tuned"].each{|parameter_type|
    out_folder = out_folder_names[de_alg]

    if parameter_type == "smac_tuned"
      out_folder = "T-" + out_folder
    end

    config_file = "de_configs/#{de_alg}_#{parameter_type}_config.dat"
    f = File.open(config_file, "r")
    paramstring = f.readline()
    f.close()

    system("python de_bbob.py -de_alg #{de_alg} -out_folder #{out_folder} #{paramstring}")
  }
}
