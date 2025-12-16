package me.crash;

import org.bukkit.Bukkit;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;

public class CrashPlugin extends JavaPlugin implements CommandExecutor {

    @Override
    public void onEnable() {
        getCommand("crash").setExecutor(this);
        getLogger().info("CrashPlugin enabled");
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {

        if (!sender.hasPermission("crash.use")) {
            sender.sendMessage("§cGeen permissie.");
            return true;
        }

        if (args.length != 1) {
            sender.sendMessage("§cGebruik: /crash <player>");
            return true;
        }

        Player target = Bukkit.getPlayer(args[0]);

        if (target == null) {
            sender.sendMessage("§cSpeler niet online.");
            return true;
        }

        target.kickPlayer("§cInternal Exception: io.netty.handler.codec.DecoderException");

        sender.sendMessage("§a" + target.getName() + " is gecrasht.");
        return true;
    }
}
